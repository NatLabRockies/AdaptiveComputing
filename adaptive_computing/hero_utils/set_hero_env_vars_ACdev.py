import os

def set_hero_env_vars():
    os.environ["HERO_ENV"] = "dev"

    # Adaptive computing environment variables
    os.environ["HERO_PROJECT"] = "adaptive-computing-app"
    os.environ["HERO_CLIENT_ID"] = "f4om7c738a1um7fgjao6msve7"
    os.environ["HERO_CLIENT_SECRET"] = "mbk361rg0eedkd6k34t5cujukl19clbv50qnteqi829gnpufkde"
    os.environ["HERO_QUEUE"] = "queue-degrees-kgriffin"

    ## os.environ["HERO_QUEUE_VISIBILITY_TIMEOUT"] = "60"
    ## os.environ["HERO_DATABASE_PASSWORD"] = "8fc2a2e2-ed9e-413d-996a-72da94e11c5c"
    
    # Rental car model environment variables
    #os.environ["HERO_PROJECT"] = "aeroportal-app"
    #os.environ["HERO_CLIENT_ID"] = "1c5ngb6o6lvtdfkus0sflstdq4"
    #os.environ["HERO_CLIENT_SECRET"] = "102hhfk1bdvc7cu307s15ljkda56hhc09qh4cp3b9hj6juhf5hap"
    #os.environ["HERO_QUEUE"] = os.environ["HERO_ENV"]+"-rental-car-worker"
    #os.environ["HERO_WORKER_NAME"] = os.environ["HERO_ENV"]+"-rental-car-worker"

if __name__ == "__main__":
    set_hero_env_vars()
