"""
Config sınıfları için unit ve validation testleri
pytest ile çalıştırın: pytest test_config.py -v
"""

import pytest
import torch
from pathlib import Path
from config import Config, FastTestConfig, SmallDataConfig, LargeDataConfig, SwinConfig


class TestConfigBasics:
    """Config sınıfının temel özelliklerini test eder"""

    def test_data_root_is_path(self):
        """DATA_ROOT bir Path objesi olmalı"""
        assert isinstance(Config.DATA_ROOT, Path)

    def test_train_classes_structure(self):
        """TRAIN_CLASSES dictionary yapısında olmalı ve 4 sınıf içermeli"""
        assert isinstance(Config.TRAIN_CLASSES, dict)
        assert len(Config.TRAIN_CLASSES) == 4
        assert all(isinstance(k, int) for k in Config.TRAIN_CLASSES.keys())
        assert all(isinstance(v, str) for v in Config.TRAIN_CLASSES.values())

    def test_test_classes_structure(self):
        """TEST_CLASSES dictionary yapısında olmalı"""
        assert isinstance(Config.TEST_CLASSES, dict)
        assert len(Config.TEST_CLASSES) == 4

    def test_approach_value(self):
        """APPROACH geçerli bir değer olmalı"""
        valid_approaches = ["bilateral", "multi-view"]
        assert Config.APPROACH in valid_approaches

    def test_fusion_type_value(self):
        """FUSION_TYPE geçerli bir değer olmalı"""
        valid_fusion_types = ["early", "late", "attention"]
        assert Config.FUSION_TYPE in valid_fusion_types

    def test_num_classes_matches_train_classes(self):
        """NUM_CLASSES, TRAIN_CLASSES ile eşleşmeli"""
        assert Config.NUM_CLASSES == len(Config.TRAIN_CLASSES)

    def test_device_is_torch_device(self):
        """DEVICE bir torch.device objesi olmalı"""
        assert isinstance(Config.DEVICE, torch.device)

    def test_checkpoint_dir_is_path(self):
        """CHECKPOINT_DIR bir Path objesi olmalı"""
        assert isinstance(Config.CHECKPOINT_DIR, Path)


class TestConfigValidation:
    """Config değerlerinin mantıksal tutarlılığını test eder"""

    def test_img_size_positive(self):
        """IMG_SIZE pozitif olmalı"""
        assert Config.IMG_SIZE > 0

    def test_img_size_reasonable(self):
        """IMG_SIZE makul bir aralıkta olmalı (32-1024)"""
        assert 32 <= Config.IMG_SIZE <= 1024

    def test_batch_size_positive(self):
        """BATCH_SIZE pozitif olmalı"""
        assert Config.BATCH_SIZE > 0

    def test_num_epochs_positive(self):
        """NUM_EPOCHS pozitif olmalı"""
        assert Config.NUM_EPOCHS > 0

    def test_weight_decay_non_negative(self):
        """WEIGHT_DECAY negatif olmamalı"""
        assert Config.WEIGHT_DECAY >= 0

    def test_weight_decay_reasonable(self):
        """WEIGHT_DECAY çok büyük olmamalı"""
        assert Config.WEIGHT_DECAY < 1.0

    def test_patience_positive(self):
        """PATIENCE pozitif olmalı"""
        assert Config.PATIENCE > 0

    def test_in_channels_valid(self):
        """IN_CHANNELS 1 veya 3 olmalı"""
        assert Config.IN_CHANNELS in [1, 3]

    def test_normalize_mean_length_matches_channels(self):
        """NORMALIZE_MEAN uzunluğu IN_CHANNELS ile eşleşmeli"""
        assert len(Config.NORMALIZE_MEAN) == Config.IN_CHANNELS

    def test_normalize_std_length_matches_channels(self):
        """NORMALIZE_STD uzunluğu IN_CHANNELS ile eşleşmeli"""
        assert len(Config.NORMALIZE_STD) == Config.IN_CHANNELS

    def test_label_smoothing_factor_range(self):
        """LABEL_SMOOTHING_FACTOR 0 ile 1 arasında olmalı"""
        if Config.USE_LABEL_SMOOTHING:
            assert 0 <= Config.LABEL_SMOOTHING_FACTOR <= 1

    def test_oversampling_threshold_range(self):
        """OVERSAMPLING_THRESHOLD 0 ile 1 arasında olmalı"""
        assert 0 < Config.OVERSAMPLING_THRESHOLD <= 1

    def test_oversampling_strategy_valid(self):
        """OVERSAMPLING_STRATEGY geçerli bir değer olmalı"""
        valid_strategies = ["auto", "threshold", "manual"]
        assert Config.OVERSAMPLING_STRATEGY in valid_strategies

    def test_num_workers_non_negative(self):
        """NUM_WORKERS negatif olmamalı"""
        assert Config.NUM_WORKERS >= 0

    def test_gradcam_samples_positive(self):
        """GRADCAM_SAMPLES pozitif olmalı"""
        if Config.USE_GRADCAM:
            assert Config.GRADCAM_SAMPLES > 0

    def test_seed_is_int(self):
        """SEED integer olmalı"""
        assert isinstance(Config.SEED, int)


class TestModelLearningRates:
    """Model learning rate metodunu test eder"""

    def test_get_learning_rate_for_known_model(self):
        """Bilinen bir model için learning rate döndürmeli"""
        lr = Config.get_learning_rate('efficientnet_b0')
        assert lr == 1e-4

    def test_get_learning_rate_for_unknown_model(self):
        """Bilinmeyen bir model için default learning rate döndürmeli"""
        lr = Config.get_learning_rate('unknown_model')
        assert lr == 1e-4

    def test_all_learning_rates_positive(self):
        """Tüm learning rateler pozitif olmalı"""
        for lr in Config.MODEL_LEARNING_RATES.values():
            assert lr > 0

    def test_all_learning_rates_reasonable(self):
        """Tüm learning rateler makul aralıkta olmalı"""
        for lr in Config.MODEL_LEARNING_RATES.values():
            assert 1e-6 <= lr <= 1e-2

    def test_current_model_in_learning_rates(self):
        """Seçilen model, MODEL_LEARNING_RATES içinde olmalı"""
        assert Config.MODEL_NAME in Config.MODEL_LEARNING_RATES


class TestFastTestConfig:
    """FastTestConfig sınıfını test eder"""

    def test_inherits_from_config(self):
        """FastTestConfig, Config'den türemeli"""
        assert issubclass(FastTestConfig, Config)

    def test_reduced_epochs(self):
        """FastTestConfig daha az epoch kullanmalı"""
        assert FastTestConfig.NUM_EPOCHS < Config.NUM_EPOCHS
        assert FastTestConfig.NUM_EPOCHS == 2

    def test_smaller_batch_size(self):
        """FastTestConfig daha küçük batch size kullanmalı"""
        assert FastTestConfig.BATCH_SIZE <= Config.BATCH_SIZE
        assert FastTestConfig.BATCH_SIZE == 4

    def test_gradcam_disabled(self):
        """FastTestConfig'de Grad-CAM kapalı olmalı"""
        assert FastTestConfig.USE_GRADCAM is False

    def test_oversampling_disabled(self):
        """FastTestConfig'de oversampling kapalı olmalı"""
        assert FastTestConfig.USE_OVERSAMPLING is False


class TestSmallDataConfig:
    """SmallDataConfig sınıfını test eder"""

    def test_inherits_from_config(self):
        """SmallDataConfig, Config'den türemeli"""
        assert issubclass(SmallDataConfig, Config)

    def test_has_unique_experiment_name(self):
        """SmallDataConfig farklı experiment adı kullanmalı"""
        assert SmallDataConfig.MLFLOW_EXPERIMENT_NAME != Config.MLFLOW_EXPERIMENT_NAME
        assert "prototype" in SmallDataConfig.MLFLOW_EXPERIMENT_NAME.lower()

    def test_oversampling_enabled(self):
        """SmallDataConfig'de oversampling açık olmalı"""
        assert SmallDataConfig.USE_OVERSAMPLING is True


class TestLargeDataConfig:
    """LargeDataConfig sınıfını test eder"""

    def test_inherits_from_config(self):
        """LargeDataConfig, Config'den türemeli"""
        assert issubclass(LargeDataConfig, Config)

    def test_larger_batch_size(self):
        """LargeDataConfig daha büyük batch size kullanmalı"""
        assert LargeDataConfig.BATCH_SIZE >= Config.BATCH_SIZE

    def test_more_epochs(self):
        """LargeDataConfig daha fazla epoch kullanmalı"""
        assert LargeDataConfig.NUM_EPOCHS >= Config.NUM_EPOCHS

    def test_production_experiment_name(self):
        """LargeDataConfig production experiment adı kullanmalı"""
        assert "production" in LargeDataConfig.MLFLOW_EXPERIMENT_NAME.lower()

    def test_class_weights_enabled(self):
        """LargeDataConfig'de class weights açık olmalı"""
        assert LargeDataConfig.USE_CLASS_WEIGHTS is True

    def test_oversampling_strategy(self):
        """LargeDataConfig threshold-based oversampling kullanmalı"""
        assert LargeDataConfig.OVERSAMPLING_STRATEGY == "threshold"


class TestSwinConfig:
    """SwinConfig sınıfını test eder"""

    def test_inherits_from_config(self):
        """SwinConfig, Config'den türemeli"""
        assert issubclass(SwinConfig, Config)

    def test_correct_model_name(self):
        """SwinConfig doğru model adını kullanmalı"""
        assert "swin" in SwinConfig.MODEL_NAME.lower()

    def test_model_in_learning_rates(self):
        """Swin model adı MODEL_LEARNING_RATES içinde olmalı"""
        assert SwinConfig.MODEL_NAME in Config.MODEL_LEARNING_RATES

    def test_batch_size_adjusted(self):
        """SwinConfig batch size'ı ayarlamış olmalı"""
        assert SwinConfig.BATCH_SIZE > 0


class TestConfigConsistency:
    """Config sınıfları arasındaki tutarlılığı test eder"""

    def test_all_configs_have_device(self):
        """Tüm config sınıfları DEVICE özelliğine sahip olmalı"""
        configs = [Config, FastTestConfig, SmallDataConfig, LargeDataConfig, SwinConfig]
        for config in configs:
            assert hasattr(config, 'DEVICE')

    def test_all_configs_have_num_classes(self):
        """Tüm config sınıfları NUM_CLASSES özelliğine sahip olmalı"""
        configs = [Config, FastTestConfig, SmallDataConfig, LargeDataConfig, SwinConfig]
        for config in configs:
            assert hasattr(config, 'NUM_CLASSES')
            assert config.NUM_CLASSES == 4

    def test_all_configs_have_get_learning_rate(self):
        """Tüm config sınıfları get_learning_rate metoduna sahip olmalı"""
        configs = [Config, FastTestConfig, SmallDataConfig, LargeDataConfig, SwinConfig]
        for config in configs:
            assert hasattr(config, 'get_learning_rate')
            assert callable(config.get_learning_rate)


class TestPrintConfig:
    """print_config metodunu test eder"""

    def test_print_config_exists(self):
        """print_config metodu var olmalı"""
        assert hasattr(Config, 'print_config')
        assert callable(Config.print_config)

    def test_print_config_runs_without_error(self, capsys):
        """print_config hatasız çalışmalı"""
        try:
            Config.print_config()
            captured = capsys.readouterr()
            assert len(captured.out) > 0
        except AttributeError as e:
            pytest.fail(f"print_config failed with AttributeError: {e}")

    def test_print_config_contains_key_info(self, capsys):
        """print_config önemli bilgileri içermeli"""
        Config.print_config()
        captured = capsys.readouterr()
        output = captured.out

        assert "Model:" in output
        assert "Batch Size:" in output
        assert "Epochs:" in output
        assert "Device:" in output


if __name__ == "__main__":
    # Testleri çalıştırmak için: python test_config.py
    pytest.main([__file__, "-v", "--tb=short"])
