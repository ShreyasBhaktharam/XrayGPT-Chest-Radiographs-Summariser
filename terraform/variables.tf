variable "vm_name" {
  description = "Name of the virtual machine"
  type        = string
}

variable "flavor_name" {
  description = "Flavor to use for the virtual machine"
  type        = string
}

variable "image_name" {
  description = "Image to use for the virtual machine"
  type        = string
}

variable "key_pair_name" {
  description = "SSH key pair name"
  type        = string
}

variable "security_groups" {
  description = "Security groups to attach to the virtual machine"
  type        = list(string)
}

variable "network_name" {
  description = "Network to attach to the virtual machine"
  type        = string
}