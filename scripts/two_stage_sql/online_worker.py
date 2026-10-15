"""Isolated ASGI workers with parent-enforced, killable request deadlines."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
from pathlib import Path
import threading
import time

from scripts.two_stage_sql.modeling import Meter


class OnlineMeter(Meter):
    """Durably count starts as well as completed calls, including killed requests."""
    def invoke(self, request, native):
        event_path = self.sink.with_name(self.sink.name.replace('calls-', 'events-'))
        sequence = len(self.records)+1
        with event_path.open('a') as stream:
            stream.write(json.dumps({'event':'started','sequence':sequence,'stage':self.stage})+'\n')
        return super().invoke(request,native)


class RequestDeadline(TimeoutError):
    pass


def stop_process(process):
    if process.is_alive():
        process.terminate()
    process.join(2)
    if process.is_alive():
        process.kill()
        process.join(2)


def receive_response(connection, process, timeout_s):
    complete = threading.Event()
    values, errors = [], []

    def receive():
        try:
            values.append(connection.recv())
        except Exception as exc:
            errors.append(exc)
        finally:
            complete.set()

    receiver = threading.Thread(target=receive,daemon=True)
    receiver.start()
    if not complete.wait(timeout_s):
        stop_process(process)
        connection.close()
        receiver.join(2)
        raise RequestDeadline('request_deadline_exceeded_worker_terminated')
    if errors:
        raise errors[0]
    return values[0]


def worker_main(connection, arm, db_path, sink):
    # This process receives question text only; no case IDs/labels/Oracle output.
    from scripts.provider_config import load_provider_config
    from scripts.model_adapter import OpenAICompatibleModel
    from httpx import AsyncClient, ASGITransport
    try:
        config = load_provider_config()
        model = OnlineMeter(OpenAICompatibleModel(config=config),Path(sink))
        if arm == 'A':
            import scripts.knowledge_search, scripts.tracing
            scripts.knowledge_search.configured_search = lambda **kwargs: None
            scripts.tracing.configured_tracer = lambda *args, **kwargs: None
            from app.main import create_app
            application = create_app(db_path=Path(db_path),model=model,provider_config=config,routing_mode='model',ask_total_timeout_seconds=60,ask_model_timeout_seconds=15,ask_tool_timeout_seconds=2)
            route = '/agent/answer'
        elif arm == 'H':
            from scripts.two_stage_sql.online import create_hybrid_app
            application = create_hybrid_app(Path(db_path),model)
            route = '/experiment/structured/answer'
        else:
            raise ValueError('invalid_arm')

        async def serve():
            async with AsyncClient(transport=ASGITransport(app=application),base_url='http://experiment') as client:
                connection.send({'ready':True,'unified_loop_enabled':config.unified_agent_loop_enabled})
                while True:
                    message = connection.recv()
                    if message is None:
                        return
                    begin = len(model.records)
                    model.stage = 'agent' if arm == 'A' else 'semantic'
                    start = time.perf_counter()
                    payload = {'question':message}
                    if arm == 'A':
                        payload['routing_mode'] = 'model'
                    try:
                        response = await client.post(route,json=payload)
                        result = {'http_status':response.status_code,'response':response.json()}
                    except Exception as exc:
                        result = {'http_status':None,'response':{},'error':type(exc).__name__}
                    result.update(service_latency_ms=round((time.perf_counter()-start)*1000,3),model_requests=model.records[begin:])
                    connection.send(result)
        asyncio.run(serve())
    except Exception as exc:
        try:
            connection.send({'ready':False,'error':type(exc).__name__})
        except (BrokenPipeError,EOFError):
            pass
    finally:
        connection.close()


class WorkerClient:
    def __init__(self, arm, db_path, output):
        self.arm, self.db_path, self.output = arm, db_path, output
        self.process = None
        self.connection = None
        self.bootstraps = []
        self.generation = 0

    def start(self):
        self.generation += 1
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        self.connection = parent
        sink = self.output/f'calls-{self.arm}-{self.generation}.jsonl'
        self.process = context.Process(target=worker_main,args=(child,self.arm,str(self.db_path),str(sink)))
        started = time.perf_counter()
        self.process.start()
        child.close()
        try:
            body = receive_response(parent,self.process,30)
            if not body.get('ready'):
                raise RuntimeError(f"worker_init_failed:{body.get('error')}")
        except Exception:
            self.close()
            raise
        self.bootstraps.append({'generation':self.generation,'latency_ms':round((time.perf_counter()-started)*1000,3),'settings':body})

    def ask(self, question, *, timeout_s=60):
        if self.process is None or not self.process.is_alive():
            self.close()
            self.start()
        started = time.perf_counter()
        sink = self.output/f'calls-{self.arm}-{self.generation}.jsonl'
        def completed_calls():
            return [json.loads(line) for line in sink.read_text().splitlines()] if sink.exists() else []
        begin = len(completed_calls())
        try:
            self.connection.send(question)
            result = receive_response(self.connection,self.process,timeout_s)
        except (RequestDeadline,EOFError,BrokenPipeError) as exc:
            self.close()
            result = {'http_status':None,'response':{},'error':type(exc).__name__,'model_requests':[]}
        result['latency_ms'] = round((time.perf_counter()-started)*1000,3)
        result['model_requests'] = completed_calls()[begin:]
        result['worker_generation'] = self.generation
        return result

    def close(self):
        if self.process is not None:
            stop_process(self.process)
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
