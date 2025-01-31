from ocp_utilities.operators import install_operator
from ocp_resources.resource import get_client
from ocp_resources.catalog_source import CatalogSource
from ocp_resources.package_manifest import PackageManifest
from ocp_resources.pod import Pod

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.dsc_initialization import DSCInitialization

from io import StringIO

import yaml
import time
import logging
import sys

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

DEFAULT_REPO = "https://github.com/trustyai-explainability/trustyai-service-operator/tarball/main"
RECHECK_INTERVAL = 5


def header(text):
    logger.info("============== {} ==============".format(text))


def wait_for_catalog_sources(operator_data):
    """Make sure all requested catalog sources are available"""
    header("Waiting for Catalog Sources")

    client = get_client()
    catalog_sources = {o['catalogSource'] for o in operator_data}

    for catalog_source in catalog_sources:
        tries = 0
        while True:
            available_sources = [s.name for s in CatalogSource.get(dyn_client=client)]
            if catalog_source in available_sources:
                logger.info("{} catalog found".format(catalog_sources))
                break
            else:
                time.sleep(RECHECK_INTERVAL)

            if tries > 300 // RECHECK_INTERVAL:
                logger.error("Catalog Source {} not found".format(catalog_source))
                raise TimeoutError("Catalog Source {} not found".format(catalog_source))

            tries += 1


def wait_for_package_manifests(operator_data):
    """Make sure the package manifest for each requested operator is available"""
    header("Waiting for Package Manifests")

    client = get_client()
    previous_return_list = None
    for operator in operator_data:
        tries = 0
        found = False
        while not found:
            # cache return value for later use
            if tries == 0 and previous_return_list is not None:
                package_manifests = previous_return_list
            else:
                package_manifests = list(PackageManifest.get(dyn_client=client))
                previous_return_list = package_manifests

            for package_manifest in package_manifests:
                if package_manifest.name == operator['name']:
                    logger.info("{} package manifest found".format(operator['name']))
                    found = True
                    break

            if not found:
                time.sleep(RECHECK_INTERVAL)

            if tries > 900 // RECHECK_INTERVAL:
                logger.error("Package Manifests for {} not found".format(operator['name']))
                raise TimeoutError("Package Manifests for {} not found".format(operator['name']))

            tries += 1


def install_operators(operator_data):
    """Install the specified operator"""
    header("Installing Operators")

    client = get_client()
    for operator in operator_data:
        install_operator(
            admin_client=client,
            target_namespaces=[],
            name=operator['name'],
            channel=operator['channel'],
            source=operator['catalogSource'],
            operator_namespace=operator['namespace'],
            timeout=600,
            install_plan_approval="Manual",
            starting_csv="{}.v{}".format(operator['name'], operator['version'])
        )


def verify_operator_running(operator_data):
    """Make sure all operator pods are running"""
    header("Verifying Operator Pods")

    client = get_client()
    previous_return_list = None
    for operator in operator_data:
        for target_pod_name in operator['correspondingPods']:
            tries = 0
            found = False

            while not found:
                # cache return value for later use
                if tries == 0 and previous_return_list is not None:
                    running_pods = previous_return_list
                else:
                    running_pods = list(Pod.get(dyn_client=client, namespace=operator['namespace']))
                    previous_return_list = running_pods

                for pod in running_pods:
                    num_running_containers = sum(
                        [1 for container in pod.exists.status['containerStatuses'] if container['started']])

                    # check if pod name matches and is running
                    if target_pod_name in pod.name and num_running_containers == 1:
                        logger.info("{} pod running".format(target_pod_name))
                        found = True
                        break

                if not found:
                    time.sleep(RECHECK_INTERVAL)

                if tries > RECHECK_INTERVAL // RECHECK_INTERVAL:
                    logger.error("Timeout waiting for {} pod".format(target_pod_name))
                    raise TimeoutError("Timeout waiting for {} pod".format(target_pod_name))

                tries += 1


def install_dsci():
    """Install a default DSCI"""
    header("Installing DSCI")

    client = get_client()
    dsci = DSCInitialization(client=client, yaml_file="manifests/dsci.yaml")
    dsci.create()


def install_datascience_cluster(trustyai_manifests_url):
    """Install a DSC that uses the specified manifests url"""
    header("Installing Datascience Cluster")

    logger.info("Using manifests from {}".format(trustyai_manifests_url))
    with open("manifests/dsc_template.yaml", "r") as f:
        template = f.read()
    config = template.replace("TRUSTYAI_REPO_PLACEHOLDER", trustyai_manifests_url)

    client = get_client()
    dsc = DataScienceCluster(client=client, yaml_file=StringIO(config))
    dsc.create()


# main function
def setup_cluster(operator_config_yaml, trustyai_manifests_url):
    # load config info
    try:
        with (open(operator_config_yaml, 'r') as f):
            operator_data = yaml.load(f, yaml.Loader)
    except FileNotFoundError as e:
        logger.error(e)
        logger.error("Operator config yaml not found, make sure your working directory is trustyai-tests/")
        raise e

    # make sure cluster is ready for operator installation
    wait_for_catalog_sources(operator_data)
    wait_for_package_manifests(operator_data)

    # install prereq operators
    install_operators(operator_data)
    verify_operator_running(operator_data)

    # install and setup ODH
    install_dsci()
    install_datascience_cluster(trustyai_manifests_url)


if __name__ == "__main__":
    operator_config_yaml = "setup/operators_config.yaml"
    trustyai_manifests_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    setup_cluster(operator_config_yaml, trustyai_manifests_url)