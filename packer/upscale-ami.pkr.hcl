# Packer template — bake AMI for upscale-BE GPU workers.
# Base: Ubuntu 22.04, region ap-southeast-1.
# Installs: NVIDIA driver, Docker, nvidia-container-toolkit, docker-compose plugin.
# Optionally pre-pulls the app image and pre-downloads model weights.
#
# Usage:
#   packer init  packer/
#   packer build -var 'image_tag=ghcr.io/vuong20031591-hub/upscale-be:main' packer/

packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = ">= 1.3.0"
    }
  }
}

variable "region"            { type = string  default = "ap-southeast-1" }
variable "instance_type"     { type = string  default = "g5.xlarge" }
variable "image_tag"         { type = string  default = "" } # e.g. ghcr.io/org/upscale-be:main
variable "prefetch_weights"  { type = bool    default = true }

source "amazon-ebs" "upscale" {
  region                      = var.region
  instance_type               = var.instance_type
  ami_name                    = "upscale-be-{{timestamp}}"
  ami_description             = "Upscale-BE GPU worker (NVIDIA + Docker + nvidia-ctk)"
  ssh_username                = "ubuntu"
  associate_public_ip_address = true

  source_ami_filter {
    filters = {
      name                = "ubuntu/images/hvm-ssd-gp3/ubuntu-jammy-22.04-amd64-server-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    owners      = ["099720109477"] # Canonical
    most_recent = true
  }

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    volume_type           = "gp3"
    delete_on_termination = true
  }
}

build {
  name    = "upscale-be"
  sources = ["source.amazon-ebs.upscale"]

  # 1) System bootstrap
  provisioner "shell" {
    inline = [
      "set -eux",
      "sudo apt-get update",
      "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg jq unzip build-essential",
    ]
  }

  # 2) NVIDIA driver
  provisioner "shell" {
    inline = [
      "set -eux",
      "sudo apt-get install -y ubuntu-drivers-common",
      "sudo ubuntu-drivers install --gpgpu",
    ]
  }

  # 3) Docker + Compose plugin
  provisioner "shell" {
    inline = [
      "set -eux",
      "sudo install -m 0755 -d /etc/apt/keyrings",
      "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
      "echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable' | sudo tee /etc/apt/sources.list.d/docker.list",
      "sudo apt-get update",
      "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
      "sudo usermod -aG docker ubuntu",
    ]
  }

  # 4) NVIDIA Container Toolkit
  provisioner "shell" {
    inline = [
      "set -eux",
      "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
      "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
      "sudo apt-get update",
      "sudo apt-get install -y nvidia-container-toolkit",
      "sudo nvidia-ctk runtime configure --runtime=docker",
      "sudo systemctl restart docker",
    ]
  }

  # 5) Optional: pre-pull image
  provisioner "shell" {
    inline = [
      "set -eux",
      "if [ -n '${var.image_tag}' ]; then sudo docker pull '${var.image_tag}' || true; fi",
    ]
  }

  # 6) Optional: pre-download Real-ESRGAN weights
  provisioner "shell" {
    inline = [
      "set -eux",
      "if [ '${var.prefetch_weights}' = 'true' ]; then",
      "  sudo mkdir -p /opt/upscale/weights",
      "  sudo curl -L -o /opt/upscale/weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
      "fi",
    ]
  }

  # 7) Cleanup
  provisioner "shell" {
    inline = [
      "sudo apt-get clean",
      "sudo rm -rf /var/lib/apt/lists/*",
      "sudo cloud-init clean --logs || true",
    ]
  }
}
