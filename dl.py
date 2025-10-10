from app.services.dl_service import encode_features, predict_from_dict


def main():
    pclass = int(input("Pclass(1, 2, 3): "))
    sex = input("Sex(male/female): ")
    fare = float(input("티켓 가격(예: 7.25): "))
    embarked = input("탑승구(C:셰르부르/Q:퀸스타운/S:사우스햄프턴): ")

    features = encode_features(pclass, sex, fare, embarked)
    result = predict_from_dict(features)

    print("입력 특징:", features)
    print("예측(원핫인코딩):", result["one_hot"])
    print("예측 결과:", result["class_label"])
    print("확률:", result["probabilities"])


if __name__ == "__main__":
    main()