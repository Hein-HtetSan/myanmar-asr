"""Build only the static research site in Vercel Build Output API format."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def build(output):
    source = ROOT / 'web'
    data = json.loads((source / 'results.json').read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not isinstance(data.get('runs'), list):
        raise ValueError('Unsupported results.json schema')
    output = Path(output)
    static = output / 'static'
    if static.exists():
        shutil.rmtree(static)
    static.mkdir(parents=True)
    for name in ('index.html', 'styles.css', 'app.js', 'results.json'):
        shutil.copyfile(source / name, static / name)
    config = {'version': 3, 'routes': [{'src': '/(.*)', 'headers': {
        'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
        'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        'Cache-Control': 'public, max-age=0, must-revalidate'}, 'continue': True}]}
    (output / 'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.vercel/output')
    build(parser.parse_args().output_dir)
