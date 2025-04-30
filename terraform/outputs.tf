output "instance_name" {
  value = openstack_compute_instance_v2.xraygpt_vm.name
}

output "instance_ip" {
  value = openstack_networking_floatingip_v2.xraygpt_floating_ip.address
}

output "ssh_command" {
  value = "ssh ubuntu@${openstack_networking_floatingip_v2.xraygpt_floating_ip.address}"
}

output "grafana_url" {
  value = "http://${openstack_networking_floatingip_v2.xraygpt_floating_ip.address}:3000"
}

output "mlflow_url" {
  value = "http://${openstack_networking_floatingip_v2.xraygpt_floating_ip.address}:6000"
}

output "spark_ui_url" {
  value = "http://${openstack_networking_floatingip_v2.xraygpt_floating_ip.address}:8080"
}

output "prometheus_url" {
  value = "http://${openstack_networking_floatingip_v2.xraygpt_floating_ip.address}:9090"
}