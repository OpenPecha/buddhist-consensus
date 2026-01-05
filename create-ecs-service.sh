#!/bin/bash

# Script to create ECS Fargate service
# Usage: ./create-ecs-service.sh

set -e

CLUSTER_NAME="webuddhist-ai-cluster"
SERVICE_NAME="webuddhist-ai-service"
TASK_DEFINITION="webuddhist-ai-task"
REGION="us-east-1"

echo "Creating ECS Fargate service..."
echo "Cluster: $CLUSTER_NAME"
echo "Service: $SERVICE_NAME"
echo "Task Definition: $TASK_DEFINITION"
echo ""

# Get default VPC
echo "Finding VPC and networking resources..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)

if [ -z "$VPC_ID" ] || [ "$VPC_ID" = "None" ]; then
    echo "Error: No default VPC found."
    echo "Please specify a VPC ID:"
    read -p "VPC ID: " VPC_ID
fi

echo "Using VPC: $VPC_ID"

# Get subnets
SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[*].SubnetId' \
    --output text --region $REGION | tr '\t' ' ')

if [ -z "$SUBNET_IDS" ]; then
    echo "Error: No subnets found in VPC."
    exit 1
fi

# Convert to comma-separated list (take first 2 subnets)
SUBNET_ARRAY=($SUBNET_IDS)
SUBNET_LIST="${SUBNET_ARRAY[0]}"
if [ ${#SUBNET_ARRAY[@]} -gt 1 ]; then
    SUBNET_LIST="${SUBNET_ARRAY[0]},${SUBNET_ARRAY[1]}"
fi

echo "Using Subnets: $SUBNET_LIST"

# Get security group
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" \
    --query 'SecurityGroups[0].GroupId' \
    --output text --region $REGION)

if [ -z "$SECURITY_GROUP_ID" ] || [ "$SECURITY_GROUP_ID" = "None" ]; then
    echo "Error: No default security group found."
    echo "Please specify a security group ID:"
    read -p "Security Group ID: " SECURITY_GROUP_ID
fi

echo "Using Security Group: $SECURITY_GROUP_ID"
echo ""

# Get latest task definition revision
LATEST_TASK_DEF=$(aws ecs describe-task-definition \
    --task-definition $TASK_DEFINITION \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text --region $REGION)

echo "Using Task Definition: $LATEST_TASK_DEF"
echo ""

# Create the service
echo "Creating ECS service..."
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $LATEST_TASK_DEF \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_LIST],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
    --platform-version LATEST \
    --region $REGION

echo ""
echo "✓ Service created successfully!"
echo ""
echo "You can check the service status with:"
echo "  aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION"







