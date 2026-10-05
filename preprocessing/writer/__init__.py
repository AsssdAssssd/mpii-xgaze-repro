from .writers import H5Writer,EveWriter,MPIIWriter
WRITERS={"default":H5Writer,"mpii":MPIIWriter,"eve":EveWriter}

def get_writer(key):
    return WRITERS[key]