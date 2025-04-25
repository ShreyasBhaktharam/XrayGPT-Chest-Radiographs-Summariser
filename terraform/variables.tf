variable "instance_count" {
  description = "Number of GPU instances for training"
  type        = number
  default     = 2
}

variable "data_volume_size" {
  description = "Size of data volume in GB"
  type        = number
  default     = 1000
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "key_pair" {
  description = "SSH key pair name"
  type        = string
  default     = "xraygpt-key"
}