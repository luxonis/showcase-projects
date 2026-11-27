"""MobileSAM selection + Track-Anything (XMem) tracking on the OAK4 HTP (ONNX Runtime QNN).

Vendored from luxonis/oak-examples apps/onnxruntime-track-anything (Apache License 2.0,
https://github.com/luxonis/oak-examples/tree/main/apps/onnxruntime-track-anything):
sessions.py, mobilesam.py, xmem.py and model_assets/ are unchanged copies of
backend/src/inference/* and backend/src/model_assets/*. Models are downloaded, verified and
prepared at image build time (oakapp.toml prepare_container -> model_assets.prepare).
"""
