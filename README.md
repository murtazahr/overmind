# Overmind

<img src="misc/overmind_logo.png" alt="logo" style="width:45px;"/> 

Overmind is a fast, scalable, and highly configurable simulator for decentralized machine learning in mobile environments.

## Setting Up Overmind

To run Overmind, you'll need to set up the following components:

* Overmind Containers
* Postgres Database
* File Storage Service

Overmind can be set up on either AWS or personal workstations. In the future, we will support other cloud platforms like GCP or Azure.

### Prerequisites

To configure and run Overmind containers, install [docker](https://docs.docker.com/get-docker/), Kubernetes Command Line Tool [kubectl](https://kubernetes.io/docs/reference/kubectl/), and create resources on [AWS EKS](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html). To run Overmind containers locally, install [minicube](https://minikube.sigs.k8s.io/docs/start/). 

### Configure Overmind Containers

Build your own overmind image with docker. You would need to update the image for every change you make to your code that implements device behavior of your decentralized learning algorithm. Here, we'll proceed with pre-built algorithms. Deploy the overmind containers by updating kubernetes configuration.

```
kubectl apply -f configs/k8s/ovm.yaml
```

Check if the services are running with `kubectl get all` command.

### Kick off examplar simulation
Start off a simulation by running `driver.py` with a tag (name of the simulation) and associated configuration file.

```
python3 driver.py --tag mass_test_droppcl --cfg configs/dist_swarm/example_gossip.json
```





