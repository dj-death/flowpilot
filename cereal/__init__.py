# pylint: skip-file
import os
import capnp

CEREAL_PATH = os.path.dirname(os.path.abspath(__file__))
capnp.remove_import_hook()

OP_CAPNP = os.environ.get("OP_CAPNP", False)

if not OP_CAPNP:
    log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
    car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
else:
    log = capnp.load(os.path.join(CEREAL_PATH, "op", "log.capnp"))
    car = capnp.load(os.path.join(CEREAL_PATH, "op", "car.capnp"))

# pycapnp 2.x compatibility: from_bytes() returns a context manager, but this codebase
# predates 2.x and uses the reader directly (e.g. cereal.messaging's msg.which()).
# Restore 1.x behavior by entering the context manager and returning the reader (which
# keeps the underlying message alive via its own reference). No-op on pycapnp 1.x.
if int(getattr(capnp, "__version__", "1").split(".")[0]) >= 2:
    def _reader_from_bytes(_orig):
        def _fb(data, *args, **kwargs):
            return _orig(data, *args, **kwargs).__enter__()
        return _fb
    for _mod in (log, car):
        for _name in dir(_mod):
            _orig = getattr(getattr(_mod, _name, None), "from_bytes", None)
            if _orig is None or not callable(_orig):
                continue
            try:
                getattr(_mod, _name).from_bytes = staticmethod(_reader_from_bytes(_orig))
            except (AttributeError, TypeError):
                pass

