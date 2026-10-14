"""Public application must start without private teaching modules or GPU extras."""
import ast
from pathlib import Path
import tomllib

from app.main import create_app
from scripts.provider_config import ProviderConfig


def test_public_app_has_no_teaching_routes_or_imports():
    application = create_app(provider_config=ProviderConfig.empty())
    assert not any(route.path.startswith('/pto') for route in application.routes)
    tree = ast.parse(Path('app/main.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'app.pto_routes' for n in ast.walk(tree))


def test_gpu_dependencies_are_explicit_opt_in():
    project = tomllib.loads(Path('pyproject.toml').read_text())
    dependencies = project['project']['dependencies']
    assert not any(d.startswith(('torch', 'transformers', 'sentence-transformers', 'pytest')) for d in dependencies)
    assert project['project']['optional-dependencies']['local-embedding']
    assert any(d.startswith('pytest') for d in project['dependency-groups']['dev'])
