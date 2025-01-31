# Cluster Setup for Testing

## Running Cluster Setup
* `poetry install .`
* `poetry run python trustyai_tests/setup/setup_cluster.py `

## Usage
```
usage: TrustyAI CI Cluster Setup [-h] [--trustyai_manifests_url TRUSTYAI_MANIFESTS_URL] [--install_operators] [--install_dsc] [--artifact_dir ARTIFACT_DIR]

Configure a fresh Openshift cluster in preparation for the TrustyAI CI test suite

options:
  -h, --help            show this help message and exit
  --trustyai_manifests_url TRUSTYAI_MANIFESTS_URL
                        URL of the TrustyAI manifests tarball. Defaults to `main` if not specified.
  --install_operators   Whether to install the prerequisite operators. Set to false if they are already installed on your cluster.
  --install_dsc         Whether to install the ODH DataScienceCluster. Set to false if you already have a DataScienceCluster running.
  --artifact_dir ARTIFACT_DIR
                        Directory where test artifacts are stored.
```