# Packer AMI cho Upscale-BE

Bake AMI cho EC2 GPU worker (g5.*/g6.*) chạy `upscale-be`.

## Prereqs
- Packer >= 1.10
- AWS CLI đã cấu hình credentials (region `ap-southeast-1`)
- Quyền IAM: `AmazonEC2FullAccess` (tối thiểu cho Packer)

## Build
```bash
cd packer
packer init .
packer build \
  -var 'region=ap-southeast-1' \
  -var 'instance_type=g5.xlarge' \
  -var 'image_tag=ghcr.io/vuong20031591-hub/upscale-be:main' \
  .
```

AMI mới xuất hiện trong console với tên `upscale-be-<timestamp>`. Dùng ID này cho Launch Template / ASG.

## Nội dung AMI
- Ubuntu 22.04 LTS
- NVIDIA driver (ubuntu-drivers gpgpu)
- Docker CE + compose plugin
- nvidia-container-toolkit (đã `nvidia-ctk runtime configure`)
- (Tuỳ chọn) Docker image `upscale-be` đã pull sẵn
- (Tuỳ chọn) Weights `RealESRGAN_x4plus.pth` tại `/opt/upscale/weights/`

## First boot trên EC2
User-data đề xuất:
```bash
#!/bin/bash
set -eux
cd /opt/upscale
cat > docker-compose.yml <<'YAML'
# copy nội dung docker-compose.yml của repo, gắn /opt/upscale/weights vào /app/weights
YAML
docker compose up -d
```
