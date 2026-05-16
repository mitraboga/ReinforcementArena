# Streamlit Community Cloud Deployment

## App Entry Point

Use this file as the Streamlit app entry point:

```text
app/streamlit_app.py
```

## Dependencies

Streamlit Community Cloud installs `requirements.txt`. This file intentionally contains only the dashboard/runtime dependencies:

```text
numpy
pandas
matplotlib
pyyaml
streamlit
```

PyTorch is optional and is kept in `requirements-ml.txt` because the deployed dashboard does not need to train DQN models at startup.

## Artifact Behavior

The dashboard loads generated CSV artifacts from `artifacts/` when they exist. If artifacts are not present in the deployed repository, the app falls back to built-in demo metrics so the public link still opens cleanly.

## Recommended Cloud Settings

- Main file path: `app/streamlit_app.py`
- Python version: 3.12
- Requirements file: `requirements.txt`

## Running On A VM Or EC2 Instance

If serving from a public VM IP instead of Streamlit Community Cloud, start the app with:

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

The VM firewall and cloud security group must allow inbound TCP traffic on port `8501`.

For AWS EC2:

- Security group inbound rule: `Custom TCP`, port `8501`, source `0.0.0.0/0` for public demo access or your own IP for restricted access.
- Instance firewall must also allow port `8501`.
- The Streamlit process must keep running after the SSH session closes, for example with `tmux`, `screen`, `nohup`, or a systemd service.
- Use `http://PUBLIC_IP:8501`, not `https://PUBLIC_IP:8501`, unless a reverse proxy and TLS certificate are configured.
