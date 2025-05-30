def get_machine_name():
    import socket
    hostname = socket.gethostname()
    if hostname.startswith("kl"): # kestrel login node
        machine_name = "kestrel"
    elif hostname.startswith("vs"): # vermilion login or comput node
        machine_name = "vermilion"
    elif hostname.startswith("x"): # kestrel compute node
        machine_name = "kestrel"
        #machine_name = hostname -f | cut -d '.' -f4 # this works for kestrel compute nodes.
    else:
        machine_name = hostname  # Default to the hostname if no match
    print(f'The machine name is {machine_name}.')
    return machine_name
    
if __name__ == "__main__":
    get_machine_name()
