# Cluster Setup for Testing

## Running Cluster Setup
* `poetry install .`
* `poetry run python trustyai_tests/setup/setup_cluster.py `

## Usage
```
usage: TrustyAI CI Cluster Setup [-h] [--trustyai_manifests_url TRUSTYAI_MANIFESTS_URL] [--skip_operator_installation] [--skip_dsc_installation] [--artifact_dir ARTIFACT_DIR]

Configure a fresh Openshift cluster in preparation for the TrustyAI CI test suite

options:
  -h, --help            show this help message and exit
  --trustyai_manifests_url TRUSTYAI_MANIFESTS_URL
                        URL of the TrustyAI manifests tarball. Defaults to `main` if not specified.
  --skip_operator_installation
                        Whether to skip the installation the prerequisite operators. Use this flag if they are already installed on your cluster.
  --skip_dsc_installation
                        Whether to skip the installation of the ODH DataScienceCluster. Use this flag if you already have a DataScienceCluster running.
  --artifact_dir ARTIFACT_DIR
                        Directory where test artifacts are stored.