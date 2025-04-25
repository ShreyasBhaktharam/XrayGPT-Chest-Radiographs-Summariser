terraform {
  required_providers {
    openstack = {
      source = "terraform-provider-openstack/openstack"
      version = "~> 1.48.0"
    }
  }
  backend "s3" {
    bucket = "xraygpt-terraform-state"
    key    = "chameleon/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "openstack" {
  cloud = "chameleon"
}

# Network resources
resource "openstack_networking_network_v2" "xraygpt_network" {
  name           = "xraygpt-network"
  admin_state_up = true
}

resource "openstack_networking_subnet_v2" "xraygpt_subnet" {
  name       = "xraygpt-subnet"
  network_id = openstack_networking_network_v2.xraygpt_network.id
  cidr       = "192.168.1.0/24"
  ip_version = 4
}

# Security groups
resource "openstack_networking_secgroup_v2" "xraygpt_sg" {
  name        = "xraygpt-sg"
  description = "Security group for XrayGPT"
}

resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

resource "openstack_networking_secgroup_rule_v2" "http" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 80
  port_range_max    = 80
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

# Compute instances
resource "openstack_compute_instance_v2" "data_pipeline" {
  name            = "xraygpt-data-pipeline"
  image_name      = "Ubuntu-20.04"
  flavor_name     = "m1.large"
  key_pair        = "xraygpt-key"
  security_groups = [openstack_networking_secgroup_v2.xraygpt_sg.name]

  network {
    uuid = openstack_networking_network_v2.xraygpt_network.id
  }
}

resource "openstack_compute_instance_v2" "training_gpu" {
  count           = 2
  name            = "xraygpt-training-${count.index}"
  image_name      = "Ubuntu-20.04"
  flavor_name     = "gpu_a10"
  key_pair        = "xraygpt-key"
  security_groups = [openstack_networking_secgroup_v2.xraygpt_sg.name]

  network {
    uuid = openstack_networking_network_v2.xraygpt_network.id
  }
}

# Storage volumes
resource "openstack_blockstorage_volume_v3" "data_volume" {
  name        = "xraygpt-data"
  size        = 1000
  description = "Persistent storage for datasets and models"
}

resource "openstack_compute_volume_attach_v2" "data_attachment" {
  instance_id = openstack_compute_instance_v2.data_pipeline.id
  volume_id   = openstack_blockstorage_volume_v3.data_volume.id
}

# Floating IPs
resource "openstack_networking_floatingip_v2" "api_ip" {
  pool = "public"
}

resource "openstack_compute_floatingip_associate_v2" "api_associate" {
  floating_ip = openstack_networking_floatingip_v2.api_ip.address
  instance_id = openstack_compute_instance_v2.data_pipeline.id
}