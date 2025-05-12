# Create a security group for XrayGPT services
resource "openstack_networking_secgroup_v2" "xraygpt_sg" {
  name        = "xraygpt-sg"
  description = "Security group for XrayGPT services"
}

# Allow SSH access
resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

# Allow access to Grafana
resource "openstack_networking_secgroup_rule_v2" "grafana" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 3000
  port_range_max    = 3000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

# Allow access to MLflow
resource "openstack_networking_secgroup_rule_v2" "mlflow" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 6000
  port_range_max    = 6000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

# Allow access to Spark UI
resource "openstack_networking_secgroup_rule_v2" "spark" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8080
  port_range_max    = 8080
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

# Allow access to Prometheus
resource "openstack_networking_secgroup_rule_v2" "prometheus" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 9090
  port_range_max    = 9090
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

# Allow access to API
resource "openstack_networking_secgroup_rule_v2" "api" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8000
  port_range_max    = 8000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.xraygpt_sg.id
}

# Create a block storage volume for data persistence
resource "openstack_blockstorage_volume_v3" "xraygpt_data" {
  name        = "xraygpt-data"
  description = "Data volume for XrayGPT services"
  size        = 100 
}

resource "openstack_compute_volume_attach_v2" "xraygpt_data_attach" {
  instance_id = openstack_compute_instance_v2.xraygpt_vm.id
  volume_id   = openstack_blockstorage_volume_v3.xraygpt_data.id
}

# Create the virtual machine
resource "openstack_compute_instance_v2" "xraygpt_vm" {
  name            = var.vm_name
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  key_pair        = var.key_pair_name
  security_groups = concat(var.security_groups, [openstack_networking_secgroup_v2.xraygpt_sg.name])

  network {
    name = var.network_name
  }

  # Bootstrap script to install Docker and Docker Compose
  user_data = <<-EOF
    #!/bin/bash
    # Update system
    apt-get update
    apt-get upgrade -y
    
    # Install dependencies
    apt-get install -y ca-certificates curl gnupg lsb-release git
    
    # Add Docker's official GPG key
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Install Docker Compose
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    # Add ubuntu user to docker group
    usermod -aG docker ubuntu
    
    # Create directory for XrayGPT project
    mkdir -p /opt/xraygpt
    chown -R ubuntu:ubuntu /opt/xraygpt
    
    # Create prometheus config file
    mkdir -p /opt/xraygpt/k8s/monitoring
    cat > /opt/xraygpt/k8s/monitoring/prometheus.yaml << 'PROMCONFIG'
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']
    PROMCONFIG
    
    # Create docker-compose.yml
    cat > /opt/xraygpt/docker-compose.yml << 'DOCKERCOMPOSE'
    version: '3.8'
    
    services:
      postgres:
        image: postgres:13-alpine
        environment:
          POSTGRES_DB: xraygpt
          POSTGRES_USER: xraygpt
          POSTGRES_PASSWORD: password
        ports:
          - "5432:5432"
        volumes:
          - postgres_data:/var/lib/postgresql/data
        networks:
          - xraygpt-network
    
      zookeeper:
        image: bitnami/zookeeper:3.8
        environment:
          - ALLOW_ANONYMOUS_LOGIN=yes
        ports:
          - "2181:2181"
        networks:
          - xraygpt-network
    
      kafka:
        image: bitnami/kafka:3.4
        depends_on:
          - zookeeper
        ports:
          - "9092:9092"
        environment:
          - KAFKA_CFG_ZOOKEEPER_CONNECT=zookeeper:2181
          - ALLOW_PLAINTEXT_LISTENER=yes
          - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092
          - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
        networks:
          - xraygpt-network
    
      spark:
        image: bitnami/spark:3.3
        environment:
          - SPARK_MODE=master
        ports:
          - "8080:8080"
          - "7077:7077"
        networks:
          - xraygpt-network
    
      mlflow:
        image: python:3.9-slim
        ports:
          - "6000:5000"  # Map container port 5000 to host port 6000
        command: >
          bash -c "pip install mlflow==2.3.0 &&
                  mkdir -p /mlflow/mlruns &&
                  cd /mlflow &&
                  mlflow ui --host 0.0.0.0"
        volumes:
          - mlflow_data:/mlflow
        networks:
          - xraygpt-network
    
      prometheus:
        image: prom/prometheus:v2.45.0
        ports:
          - "9090:9090"
        volumes:
          - ./k8s/monitoring/prometheus.yaml:/etc/prometheus/prometheus.yml:ro
          - prometheus_data:/prometheus
        command:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
        networks:
          - xraygpt-network
    
      grafana:
        image: grafana/grafana:10.0.0
        ports:
          - "3000:3000"
        depends_on:
          - prometheus
        environment:
          GF_SECURITY_ADMIN_PASSWORD: admin
          GF_SERVER_ROOT_URL: http://localhost:3000
          GF_SERVER_SERVE_FROM_SUB_PATH: "true"
        volumes:
          - grafana_data:/var/lib/grafana
        networks:
          - xraygpt-network
    
    volumes:
      postgres_data:
      mlflow_data:
      prometheus_data:
      grafana_data:
    
    networks:
      xraygpt-network:
        driver: bridge
    DOCKERCOMPOSE
    
    chown -R ubuntu:ubuntu /opt/xraygpt
    EOF
}

# Attach volume to instance
resource "openstack_compute_volume_attach_v2" "xraygpt_data_attach" {
  instance_id = openstack_compute_instance_v2.xraygpt_vm.id
  volume_id   = openstack_blockstorage_volume_v3.xraygpt_data.id
}

# Create and assign floating IP
resource "openstack_networking_floatingip_v2" "xraygpt_floating_ip" {
  pool = "public"
}

resource "openstack_compute_floatingip_associate_v2" "xraygpt_floating_ip_associate" {
  floating_ip = openstack_networking_floatingip_v2.xraygpt_floating_ip.address
  instance_id = openstack_compute_instance_v2.xraygpt_vm.id
}