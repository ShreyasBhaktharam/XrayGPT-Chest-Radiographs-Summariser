output "api_ip" {
  description = "Public IP for API endpoint"
  value       = openstack_networking_floatingip_v2.api_ip.address
}

output "data_pipeline_ip" {
  description = "Private IP of data pipeline server"
  value       = openstack_compute_instance_v2.data_pipeline.access_ip_v4
}

output "training_ips" {
  description = "Private IPs of training servers"
  value       = openstack_compute_instance_v2.training_gpu[*].access_ip_v4
}