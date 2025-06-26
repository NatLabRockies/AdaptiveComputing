def set_hero_env_vars():
    import os
    os.environ["HERO_ENV"] = "dev"
    os.environ["HERO_PROJECT"] = "adaptive-computing-app"
    os.environ["HERO_CLIENT_ID"] = <client_id> # modify this. Ask your Hero admin.
    os.environ["HERO_CLIENT_SECRET"] =  <your_client_secret> # modify this. Ask your Hero admin.
    os.environ["HERO_QUEUE"] = <your_unique_queue_name> # modify this. A unique string for your workflow. Consider using something containing your username to avoid interactions with other users.
    os.environ["HERO_QUEUE_VISIBILITY_TIMEOUT"] = "60"
    os.environ["HERO_DATABASE_PASSWORD"] = <your_database_password> # modify this. Ask your Hero admin.

if __name__ == "__main__":
    set_hero_env_vars()
