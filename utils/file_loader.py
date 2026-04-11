def load_text_file(path):
    with open(path, "r") as f:
        return f.readlines()