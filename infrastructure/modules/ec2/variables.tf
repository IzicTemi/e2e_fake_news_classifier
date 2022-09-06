
variable "bucket_name" {
    description = "Name of the bucket"
    type = string
}

variable "ami" {
    type = string
    default = "ami-052efd3df9dad4825"
}

variable "instance_type" {
    type = string
    default = "t2.micro"
}
