import torch
import matplotlib.pyplot as plt
import numpy as np
import math

## ========================================================
## 함수이름 : train_one_epoch
## 반환결과 : 손실, 정확도
## ========================================================
# 호출 인자: (model, trainDL, lossFn, optim, DEVICE)
def train_one_epoch(model, loader, loss_fn, optim, device):
    ## 모델 동작 모드 설정 : 반드시 학습 전에 서정
    model.train()

    ## 1에포크 후 학습 결과 저장용
    total_loss, correct, total = 0.0, 0, 0

    ## 배치크기만큼 학습 진행
    for x, y in loader:
        ## 피쳐, 타겟 추출 후 위치 설정 
        ## -> GPU 메모리로 복사(이동)
        x, y = x.to(device), y.to(device)

        ## grad 추적 기록 지우기
        optim.zero_grad()

        ## 순전파 forwarding method 호출
        logits = model(x)

        ## 각 배치사이즈 단위당 첫 10개 클래스에 대한 예측 raw_score
        ## print(f'\n{logits[0]}\n')

        loss = loss_fn(logits, y)

        # 역전파
        ## 가중치 기울기 계산
        loss.backward()
        ## 가중치 실제 업데이트
        optim.step()

        # 학습 결과 계산 및 저장 
        ## 평균손실값 * 배치사이즈
        total_loss += loss.item() * x.size(0)

        ## 열방향으로 가장 큰 값의 클래스 인덱스를 뽑음!
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


## ========================================================
## 함수이름 : evaluate
## 함수기능 : 검증/테스트 시 사용
##           역전파 진행 하지 않음!!!
## 반환결과 : 손실, 정확도
## ========================================================
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    ## 모델 동작 모드 => 학습에 필수적인 기능들 비활성화
    model.eval()

    ## 검증 시 결과 저장 변수들
    total_loss, correct, total = 0.0, 0, 0

    ## 검증/테스트용 배치크기만큼 로딩 후 예측 진행
    for x, y in loader:
        ## 피쳐, 타겟 추출 후 위치 설정 
        ## -> GPU 메모리로 복사(이동)
        x, y = x.to(device), y.to(device)

        ## 순전파
        logits = model(x)
        loss = loss_fn(logits, y)

        ## 예측 결과 저장 
        total_loss += loss.item() * x.size(0)
        pred        = logits.argmax(dim=1)
        correct    += (pred == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total

## ========================================================
## 함수이름 : train_one_epoch_cat 카테고리 포함
## 반환결과 : 손실, 정확도
## ========================================================
def train_one_epoch_cat(model, loader, loss_fn, optim, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for (x_num, x_cat), y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        logits = model((x_num, x_cat))
        loss = loss_fn(logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

## ========================================================
## 함수이름 : evaluate_cat 카테고리 포함
## 함수기능: 검증/테스트 시 사용
##          역전파 진행하지 않음
## 반환결과 : 손실, 정확도
## ========================================================
## 데코레이터, 미리 알려주는 역할
@torch.no_grad()
def evaluate_cat(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for (x_num, x_cat), y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        logits = model((x_num, x_cat))
        loss = loss_fn(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


## ========================================================
## 함수이름 : show_predictions
## 반환결과 :  - 
## ========================================================
@torch.no_grad()
def show_predictions(model, loader, class_names, device, n=8):
    model.eval()

    x, y = next(iter(loader))
    x = x.to(device)

    logits = model(x)
    ## 예측값과 정답값을 다시 cpu로 가지고 옴
    pred = logits.argmax(dim=1).cpu()
    y = y.cpu()

    n = min(n, x.size(0))
    print("Sample predictions:")
    for i in range(n):
        gt   = class_names[y[i].item()]
        pd   = class_names[pred[i].item()]
        mark = "[O]" if gt == pd else "[X]"
        print(f"- GT: {gt:<5} | Pred: {pd:<5} {mark}")

## ========================================================
## 함수이름 : show_predictions_all
## 반환결과 :  -
## ========================================================

@torch.no_grad()
def show_predictions_all(model, loader, class_names, device):
    model.eval()

    print("All test predictions:")
    for x, y in loader:
        x = x.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1).cpu()
        y = y.cpu()

        for i in range(len(y)):
            gt   = class_names[y[i].item()]
            pd   = class_names[pred[i].item()]
            mark = "[O]" if gt == pd else "[X]"
            print(f"- GT: {gt:<5} | Pred: {pd:<5} {mark}")

## ========================================================
## 함수이름 : plot_history
## 반환결과 :  - 
## ========================================================
def plot_history(history, title="Training Curves"):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["valid_loss"],   label="valid_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="train_acc (%)")
    plt.plot(epochs, [a * 100 for a in history["valid_acc"]],   label="valid_acc (%)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"{title} - Accuracy")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()


## ======================== [회귀용 함수]
def train_one_epoch_reg(model, loader, lossFn, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / n

@torch.no_grad()
def evaluate_reg(model, loader, lossFn, device):
    model.eval()
    total_loss, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = lossFn(pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / n

@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    preds, trues = [], []

    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy().reshape(-1)
        true = y.numpy().reshape(-1)
        preds.append(pred)
        trues.append(true)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mse = np.mean((preds - trues) ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(preds - trues))
    return rmse, mae

def plot_history_reg(history, title="Training Curves (Regression)"):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


## ================================================================
## [이미지용]
def train_one_epoch_img(model, loader, loss_fn, optim, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad()
        logits = model(x)              # (B, 10)
        loss = loss_fn(logits, y)

        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_img(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader: 
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

def train_eval_full(model, train_loader, val_loader, test_loader, loss_fn, optim, device, epochs):
    # 1. 결과 저장용 딕셔너리
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': []
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- [Train] ---
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optim, device)
        
        # --- [Evaluation] ---
        valid_loss, valid_acc = evaluate(model, val_loader, loss_fn, device)

        # 기록 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        # Best Model 체크포인트 (Deep Copy 대체)
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            # state_dict를 안전하게 복사하여 저장
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f'[EPOCH-{epoch:03}] TRAIN => Loss: {train_loss:.7f} Acc: {train_acc:.5f}')
        print(f'{" " * 11} VALID => Loss: {valid_loss:.7f} Acc: {valid_acc:.5f}')

    # 최고 성능 가중치로 복원
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n★ Best Validation Accuracy: {best_val_acc*100:.2f}% (Restored)")

    # 최종 테스트 (학습 완료 후 딱 한 번)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
    print(f" 최종 TEST 결과 => Loss: {te_loss:.4f} Acc: {te_acc*100:.2f}%")

    return history