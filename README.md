docker build -t fixai-api .
docker run -d -p 8011:8011 --name fixai-container fixai-api