"""Persistent local Demo profile; generic application configuration is unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE = ROOT / '.data/ecommerce/demo.env'


def configure(profile=PROFILE):
    from dotenv import dotenv_values, load_dotenv
    if not profile.exists():
        raise RuntimeError('Run scripts/start_ecommerce_demo.py --init-config first')
    values = dotenv_values(profile)
    required = {'MYSQL_HOST', 'MYSQL_PORT', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE'}
    if not required <= values.keys() or not all(values[k] for k in required):
        raise ValueError('Demo profile is missing required MySQL settings')
    if (values['MYSQL_DATABASE'], values['MYSQL_USER']) != ('insight_ecommerce_db', 'ecommerce_ro'):
        raise ValueError('Demo profile must use the ecommerce read-only database/account')
    load_dotenv(ROOT / '.env')
    os.environ.update({k: v for k, v in values.items() if k in required})


def initialize():
    from dotenv import set_key
    if PROFILE.exists():
        print('Existing Demo profile retained: .data/ecommerce/demo.env')
        return
    credentials = json.loads((ROOT / '.data/ecommerce/ecommerce_ro.json').read_text(encoding='utf-8'))
    if (credentials['database'], credentials['user']) != ('insight_ecommerce_db', 'ecommerce_ro'):
        raise ValueError('Unexpected demo account')
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE.touch(exist_ok=False)
    for field in ('host', 'port', 'user', 'password', 'database'):
        set_key(str(PROFILE), 'MYSQL_' + field.upper(), str(credentials[field]))
    print('Created ignored local Demo profile; no credentials printed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-config', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.init_config:
        initialize()
        return
    configure()
    if args.check:
        import mysql.connector
        from app.tools.db_tools import get_db_config
        with mysql.connector.connect(**get_db_config()) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT DATABASE(), CURRENT_USER()')
                print(json.dumps({'database_identity': cursor.fetchone(),
                                 'tavily_configured': bool(os.getenv('TAVILY_API_KEY')),
                                 'deepseek_configured': bool(os.getenv('OPENAI_API_KEY'))}))
        return
    import uvicorn
    uvicorn.run('app.api.server:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
