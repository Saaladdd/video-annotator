"""Cloud/remote GPU worker for the auto-labeler.

The worker runs on a machine that has a GPU (and a working PyTorch/CUDA stack)
and exposes the same `LocalBackend` over a tiny HTTP API. The labeler client
uses `labeler.backends.remote_backend.RemoteBackend` to talk to it, so all
storage and outputs remain on the client.
"""
