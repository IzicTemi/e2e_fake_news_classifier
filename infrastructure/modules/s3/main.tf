resource "aws_s3_bucket" "s3_bucket" {
  bucket = var.bucket_name
}

output "name" {
  value = aws_s3_bucket.s3_bucket.bucket
}
