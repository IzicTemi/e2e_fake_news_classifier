resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "main" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.0.0/24"
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

}

resource "aws_route_table" "second_rt" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

}

resource "aws_route_table_association" "public_subnet" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.second_rt.id
}

resource "aws_security_group" "allow_http_ssh" {
  name        = "allow_http"
  description = "Allow http inbound traffic"
  vpc_id = aws_vpc.main.id


  ingress {
    description = "http"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]

  }
ingress {
    description = "ssh"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]

  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "allow_http_ssh"
  }
}


resource "aws_iam_instance_profile" "profile" {
  name = "mlflow_profile"
  role = aws_iam_role.role.name
}

resource "aws_iam_role" "role" {
  name = "mlflow_role"
  path = "/"

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": "sts:AssumeRole",
            "Principal": {
               "Service": "ec2.amazonaws.com"
            },
            "Effect": "Allow",
            "Sid": ""
        }
    ]
}
EOF
}

resource "aws_iam_policy" "mlflow_s3_role_policy" {
  name = "mlflow_policy"
  description = "IAM Policy for s3"
policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
            "Effect": "Allow",
            "Action": [
                "s3:ListAllMyBuckets",
                "s3:GetBucketLocation",
                "s3:*"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::${var.bucket_name}",
                "arn:aws:s3:::${var.bucket_name}/*"
            ]
        }
  ]
}
  EOF
}

resource "aws_iam_role_policy_attachment" "iam-policy-attach" {
  role       = aws_iam_role.role.name
  policy_arn = aws_iam_policy.mlflow_s3_role_policy.arn
}

resource "aws_instance" "mlflow_instance" {
  ami           = var.ami
  instance_type = var.instance_type
  subnet_id = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.allow_http_ssh.id]
  associate_public_ip_address = true
  key_name = "webserver_key"
  iam_instance_profile = aws_iam_instance_profile.profile.name

  connection {
        type    = "ssh"
        user    = "ec2-user"
        host    = self.public_ip
        port    = 22
        private_key = file("${path.module}/webserver_key.pem")
  }

  provisioner "remote-exec" {
      inline  = [
          "echo \"export ARTIFACT_LOC=${var.bucket_name}\" >> ~/.bashrc"
	    ]
  }

  provisioner "remote-exec" {
    scripts = [
      "./scripts/startup.sh"
    ]
  }

  tags = {
    Name = "mlflow_server"
  }
}

resource "aws_ebs_volume" "task_volume" {
  availability_zone = aws_instance.mlflow_instance.availability_zone
  size              = 1
}
resource "aws_volume_attachment" "ebs_att" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.task_volume.id
  instance_id = aws_instance.mlflow_instance.id
  force_detach = true
  depends_on = [aws_ebs_volume.task_volume]
}

output "ec2_url" {
    value = "http://${aws_instance.mlflow_instance.public_ip}:5000"
}
