import yaml

data = {
    'train': '/content/dataset/dataset/train',
    'val': '/content/dataset/dataset/val',
    'nc': 1,
    'names': ['flood']
}

with open('data.yaml', 'w') as file:
    yaml.dump(data, file)

print("✅ data.yaml başarıyla oluşturuldu.")