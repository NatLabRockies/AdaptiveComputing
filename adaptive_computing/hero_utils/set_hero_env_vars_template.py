def set_hero_env_vars():
    import os
    os.environ["HERO_ENV"] = "dev"
    os.environ["HERO_PROJECT"] = "adaptive-computing-app"
    os.environ["HERO_CLIENT_ID"] = <client_id> # modify this
    os.environ["HERO_CLIENT_SECRET"] =  <you_client_secret> # modify this
    os.environ["HERO_QUEUE"] = "queue-degrees"
    os.environ["HERO_QUEUE_VISIBILITY_TIMEOUT"] = "60"
    os.environ["HERO_DATABASE_PASSWORD"] = <your_database_password> # modify this

if __name__ == "__main__":
    set_hero_env_vars()
