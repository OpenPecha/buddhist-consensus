# ECS Fargate Deployment Quick Reference

## GitHub Actions Workflow

The workflow automatically:
1. Builds Docker image on push to `fix/combine-search-chat` branch
2. Pushes to ECR Public: `public.ecr.aws/t8u7n5a4/webuddhist-ai-api:latest`
3. Deploys to ECS Fargate service

## Required GitHub Secrets

Set these in: **Repository Settings → Secrets and variables → Actions**

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## AWS Resources to Create

### 1. IAM Roles
- `ecsTaskExecutionRole` - For ECS to pull images and write logs
- `ecsTaskRole` - For application AWS service access (if needed)

### 2. Secrets Manager
Create secrets under prefix `webuddhist-ai/`:
- `GEMINI_API_KEY`
- `MILVUS_URI`
- `MILVUS_TOKEN`
- `MILVUS_COLLECTION_NAME`

### 3. CloudWatch
- Log group: `/ecs/webuddhist-ai-api`

### 4. VPC & Networking
- VPC (or use default)
- 2+ Subnets in different AZs
- Security Group: `webuddhist-ai-sg` (port 8000)
- Internet Gateway (if public access needed)

### 5. Load Balancer (Recommended)
- Application Load Balancer: `webuddhist-ai-alb`
- Target Group: `webuddhist-ai-tg` (port 8000, health check `/health`)

### 6. ECS Resources
- Cluster: `webuddhist-ai-cluster`
- Task Definition: `webuddhist-ai-task`
- Service: `webuddhist-ai-service`

## Task Definition Configuration

- **CPU:** 512 (0.5 vCPU)
- **Memory:** 1024 MB (1 GB)
- **Container Port:** 8000
- **Health Check:** `/health` endpoint

## Environment Variables

Set via Secrets Manager (see above) or directly in task definition:
- `PORT=8000`
- `ENV=production`
- `GEMINI_API_KEY` (from Secrets Manager)
- `MILVUS_URI` (from Secrets Manager)
- `MILVUS_TOKEN` (from Secrets Manager)
- `MILVUS_COLLECTION_NAME` (from Secrets Manager)

## Quick Setup Commands (AWS CLI)

```bash
# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json \
  --region us-east-1

# Update service (after task definition is registered)
aws ecs update-service \
  --cluster webuddhist-ai-cluster \
  --service webuddhist-ai-service \
  --task-definition webuddhist-ai-task \
  --region us-east-1
```

## Access Your Application

- **With ALB:** `http://ALB_DNS_NAME/health`
- **Without ALB:** `http://TASK_PUBLIC_IP:8000/health`

## Monitoring

- **Logs:** CloudWatch → Log groups → `/ecs/webuddhist-ai-api`
- **Metrics:** ECS Console → Cluster → Service → Metrics tab
- **Health:** ECS Console → Service → Tasks → Check health status

## Common Issues

1. **Tasks not starting:** Check CloudWatch logs and IAM role permissions
2. **Health check failing:** Verify `/health` endpoint and container logs
3. **Cannot access:** Check security groups and public IP assignment

For detailed setup instructions, see `AWS_SETUP_INSTRUCTIONS.md`.

