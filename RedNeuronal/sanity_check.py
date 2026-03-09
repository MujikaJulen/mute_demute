import torch
from RedNeuronal.model import EncoderWithClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderWithClassifier(hidden_size=384, num_labels=19, freeze_encoder=False)
model.to(device)
model.train()

# batch sintético
x = torch.randn(4, 3, 224, 224, device=device)
out = model(x)
print("out.shape:", out.shape)            # -> (4, 19)
print("out sum per row:", out.sum(dim=1)) # no deben sumar 1 (son logits)

# comprobar gradientes en la cabeza
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
y = torch.randint(0, 19, (4,), device=device)

for i in range(5):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.classifier.parameters() if p.grad is not None)
    print(f"iter {i}, loss={loss.item():.4f}, grad_norm={grad_norm:.6f}")
    optimizer.step()