
variable "bucket_name" {
    description = "Name of the bucket"
    type = string
}

variable "ami" {
    type = string
    default = "ami-05fa00d4c63e32376"
}

variable "instance_type" {
    type = string
    default = "t2.micro"
}
