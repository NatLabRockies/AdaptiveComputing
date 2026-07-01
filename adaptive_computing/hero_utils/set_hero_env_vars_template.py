def set_hero_env_vars():
    import os
    os.environ["HERO_ENV"] = "ask_your_admin" # modify this. Ask your Hero admin.
    os.environ["HERO_PROJECT"] = "ask_your_admin" # modify this. Ask your Hero admin.
    os.environ["HERO_CLIENT_ID"] = "ask_your_admin" # modify this. Ask your Hero admin.
    os.environ["HERO_CLIENT_SECRET"] =  "ask_your_admin" # modify this. Ask your Hero admin.
    os.environ["HERO_QUEUE"] = "your_unique_queue_name" # modify this. A unique string for your workflow. Consider using something containing your username to avoid interactions with other users on your project.

if __name__ == "__main__":
    set_hero_env_vars()
