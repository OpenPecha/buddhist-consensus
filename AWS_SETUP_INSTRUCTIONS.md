# AWS ECS Fargate Deployment Setup Instructions

This guide walks you through setting up your application on AWS ECS Fargate.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured (or use AWS Console)
- Your Docker image pushed to ECR Public (already done via GitHub Actions)

## Step-by-Step AWS Console Setup

### 1. Create IAM Roles

#### A. ECS Task Execution Role

1. Go to **IAM Console** → **Roles** → **Create role**
2. Select **AWS service** → **Elastic Container Service** → **Elastic Container Service Task**
3. Click **Next**
4. Attach the policy: **AmazonECSTaskExecutionRolePolicy**
5. Click **Next** → Give it a name: `ecsTaskExecutionRole`
6. Click **Create role**
7. **Note the ARN** (e.g., `arn:aws:iam::123456789012:role/ecsTaskExecutionRole`)

#### B. ECS Task Role (for application permissions)

1. Go to **IAM Console** → **Roles** → **Create role**
2. Select **AWS service** → **Elastic Container Service** → **Elastic Container Service Task**
3. Click **Next**
4. **Skip attaching policies** (unless you need specific AWS service access)
5. Click **Next** → Give it a name: `ecsTaskRole`
6. Click **Create role**
7. **Note the ARN** (e.g., `arn:aws:iam::123456789012:role/ecsTaskRole`)

### 2. Create Secrets in AWS Secrets Manager

1. Go to **Secrets Manager** → **Store a new secret**
2. Select **Other type of secret** → **Plaintext**
3. For each secret, create separately:

   **Secret 1: GEMINI_API_KEY**
   - Name: `webuddhist-ai/GEMINI_API_KEY`
   - Value: Your Gemini API key
   - Click **Next** → **Next** → **Store**

   **Secret 2: MILVUS_URI**
   - Name: `webuddhist-ai/MILVUS_URI`
   - Value: Your Milvus URI
   - Click **Next** → **Next** → **Store**

   **Secret 3: MILVUS_TOKEN**
   - Name: `webuddhist-ai/MILVUS_TOKEN`
   - Value: Your Milvus token
   - Click **Next** → **Next** → **Store**

   **Secret 4: MILVUS_COLLECTION_NAME**
   - Name: `webuddhist-ai/MILVUS_COLLECTION_NAME`
   - Value: `test_kangyur_tengyur` (or your collection name)
   - Click **Next** → **Next** → **Store**

4. **Note the ARN** of each secret (you'll need these for the task definition)

### 3. Create CloudWatch Log Group

1. Go to **CloudWatch** → **Log groups** → **Create log group**
2. Name: `/ecs/webuddhist-ai-api`
3. Click **Create log group**

### 4. Create VPC and Networking (if you don't have one)

#### A. Create VPC (or use default)

1. Go to **VPC Console** → **Your VPCs**
2. If you have a default VPC, note its ID
3. If not, create a new VPC:
   - Click **Create VPC**
   - Name: `webuddhist-ai-vpc`
   - IPv4 CIDR: `10.0.0.0/16`
   - Click **Create VPC**

#### B. Create Subnets

1. Go to **VPC Console** → **Subnets**
2. Create at least 2 subnets in different Availability Zones:
   - **Subnet 1:**
     - Name: `webuddhist-ai-subnet-1`
     - VPC: Select your VPC
     - Availability Zone: `us-east-1a`
     - IPv4 CIDR: `10.0.1.0/24`
   - **Subnet 2:**
     - Name: `webuddhist-ai-subnet-2`
     - VPC: Select your VPC
     - Availability Zone: `us-east-1b`
     - IPv4 CIDR: `10.0.2.0/24`

#### C. Create Internet Gateway (if needed)

1. Go to **VPC Console** → **Internet Gateways**
2. If your VPC doesn't have one:
   - Click **Create internet gateway**
   - Name: `webuddhist-ai-igw`
   - Click **Create**
   - Select it → **Actions** → **Attach to VPC** → Select your VPC

#### D. Create Security Group

1. Go to **VPC Console** → **Security Groups** → **Create security group**
2. Name: `webuddhist-ai-sg`
3. Description: `Security group for ECS Fargate service`
4. VPC: Select your VPC
5. **Inbound rules:**
   - Type: `Custom TCP`
   - Port: `8000`
   - Source: `0.0.0.0/0` (or restrict to your IP/ALB)
   - Description: `Allow HTTP traffic`
6. **Outbound rules:** Leave default (allow all)
7. Click **Create security group**
8. **Note the Security Group ID**

### 5. Create Application Load Balancer (ALB) - Recommended

1. Go to **EC2 Console** → **Load Balancers** → **Create Load Balancer**
2. Select **Application Load Balancer**
3. **Basic configuration:**
   - Name: `webuddhist-ai-alb`
   - Scheme: **Internet-facing**
   - IP address type: **IPv4**
4. **Network mapping:**
   - VPC: Select your VPC
   - Availability Zones: Select at least 2 subnets
5. **Security groups:** Select `webuddhist-ai-sg`
6. **Listeners and routing:**
   - Protocol: `HTTP`
   - Port: `80`
   - Default action: **Create target group** (we'll create this next)
7. Click **Create load balancer**
8. **Note the ARN and DNS name**

#### Create Target Group

1. Go to **EC2 Console** → **Target Groups** → **Create target group**
2. **Basic configuration:**
   - Target type: **IP addresses**
   - Name: `webuddhist-ai-tg`
   - Protocol: `HTTP`
   - Port: `8000`
   - VPC: Select your VPC
3. **Health checks:**
   - Health check path: `/health`
   - Advanced health check settings:
     - Healthy threshold: `2`
     - Unhealthy threshold: `3`
     - Timeout: `5`
     - Interval: `30`
     - Success codes: `200`
4. Click **Next** → **Create target group**
5. **Note the Target Group ARN**

### 6. Create ECS Cluster

1. Go to **ECS Console** → **Clusters** → **Create Cluster**
2. **Cluster configuration:**
   - Cluster name: `webuddhist-ai-cluster`
   - Infrastructure: **AWS Fargate (serverless)**
3. Click **Create**

### 7. Create ECS Task Definition

1. Go to **ECS Console** → **Task definitions** → **Create new task definition**
2. **Task definition family:** `webuddhist-ai-task`
3. **Launch type:** **Fargate**
4. **Operating system/Architecture:** **Linux/X86_64**
5. **Task size:**
   - CPU: `0.5 vCPU` (512)
   - Memory: `1 GB` (1024)
6. **Task execution role:** Select `ecsTaskExecutionRole`
7. **Task role:** Select `ecsTaskRole`
8. **Network mode:** **awsvpc**

9. **Container definitions** → **Add container:**
   - **Container name:** `webuddhist-ai-api`
   - **Image URI:** `public.ecr.aws/t8u7n5a4/webuddhist-ai-api:latest`
   - **Port mappings:**
     - Container port: `8000`
     - Protocol: `tcp`
   - **Environment variables:**
     - `PORT` = `8000`
     - `ENV` = `production`
   - **Secrets** (click **Add secret** for each):
     - `GEMINI_API_KEY` → Select secret: `webuddhist-ai/GEMINI_API_KEY`
     - `MILVUS_URI` → Select secret: `webuddhist-ai/MILVUS_URI`
     - `MILVUS_TOKEN` → Select secret: `webuddhist-ai/MILVUS_TOKEN`
     - `MILVUS_COLLECTION_NAME` → Select secret: `webuddhist-ai/MILVUS_COLLECTION_NAME`
   - **Logging:**
     - Log driver: **awslogs**
     - Log group: `/ecs/webuddhist-ai-api`
     - Region: `us-east-1`
     - Stream prefix: `ecs`
   - **Health check:**
     - Command: `CMD-SHELL,curl -f http://localhost:8000/health || exit 1`
     - Interval: `30`
     - Timeout: `5`
     - Start period: `60`
     - Retries: `3`
10. Click **Create**

### 8. Create ECS Service

1. Go to **ECS Console** → **Clusters** → Select `webuddhist-ai-cluster`
2. Click **Create** → **Service**
3. **Compute configuration:**
   - Launch type: **Fargate**
   - Platform version: **LATEST**
   - Operating system/Architecture: **Linux/X86_64**
4. **Deployment configuration:**
   - Task definition:
     - Family: `webuddhist-ai-task`
     - Revision: `1` (latest)
   - Service name: `webuddhist-ai-service`
   - Desired tasks: `1` (adjust as needed)
5. **Networking:**
   - VPC: Select your VPC
   - Subnets: Select at least 2 subnets
   - Security groups: Select `webuddhist-ai-sg`
   - Auto-assign public IP: **ENABLED** (if no ALB) or **DISABLED** (if using ALB)
6. **Load balancing:**
   - Load balancer type: **Application Load Balancer**
   - Load balancer name: Select `webuddhist-ai-alb`
   - Container to load balance: Select `webuddhist-ai-api:8000:8000`
   - Target group: Select `webuddhist-ai-tg`
   - Health check grace period: `60`
7. **Service auto-scaling:** (Optional)
   - Configure service auto scaling: **Enable**
   - Minimum tasks: `1`
   - Maximum tasks: `5`
   - Target CPU utilization: `70%`
8. Click **Create**

### 9. Update Security Group for ALB (if using ALB)

1. Go to **VPC Console** → **Security Groups**
2. Select the security group attached to your ALB
3. **Inbound rules** → **Edit inbound rules**
4. Add rule:
   - Type: `HTTP`
   - Port: `80`
   - Source: `0.0.0.0/0` (or restrict as needed)
5. Save rules

### 10. Update Task Definition JSON (for GitHub Actions)

1. Open `ecs-task-definition.json` in your project
2. Replace `YOUR_ACCOUNT_ID` with your AWS Account ID (12-digit number)
3. Update the secret ARNs with the actual ARNs from Secrets Manager
4. Update execution role ARN and task role ARN
5. Save the file

### 11. Register Task Definition via AWS CLI (or use console)

You can register the task definition using AWS CLI:

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json --region us-east-1
```

Or manually create it through the console as described in Step 7.

## GitHub Secrets Required

Make sure these secrets are set in your GitHub repository:

1. Go to **GitHub Repository** → **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:
   - `AWS_ACCESS_KEY_ID` - Your AWS access key ID
   - `AWS_SECRET_ACCESS_KEY` - Your AWS secret access key

## Verify Deployment

1. Go to **ECS Console** → **Clusters** → `webuddhist-ai-cluster` → **Services** → `webuddhist-ai-service`
2. Check that tasks are running (Status: **RUNNING**)
3. If using ALB, get the DNS name from Load Balancer console
4. Test the endpoint:
   - Health check: `http://ALB_DNS_NAME/health`
   - API docs: `http://ALB_DNS_NAME/docs`

## Troubleshooting

### Tasks failing to start
- Check CloudWatch logs: `/ecs/webuddhist-ai-api`
- Verify secrets are correctly configured
- Check security group allows outbound traffic
- Verify task execution role has permissions

### Health checks failing
- Ensure `/health` endpoint is accessible
- Check container logs in CloudWatch
- Verify port 8000 is correctly configured

### Cannot access service
- Verify security group allows inbound traffic on port 8000 (or 80 for ALB)
- Check that tasks have public IP (if no ALB) or are in private subnets with NAT Gateway
- Verify ALB target group health checks are passing

## Cost Optimization Tips

1. Use Fargate Spot for non-production workloads (50-70% savings)
2. Set up auto-scaling based on actual traffic patterns
3. Use CloudWatch alarms to monitor costs
4. Consider using smaller task sizes if your app doesn't need 1GB RAM

## Next Steps

- Set up CloudWatch alarms for monitoring
- Configure auto-scaling policies
- Set up CI/CD pipeline (already done via GitHub Actions)
- Configure custom domain with Route 53 and ACM certificate

