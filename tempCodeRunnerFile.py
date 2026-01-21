if __name__ == "__main__":
    combine_data(config.train_guid_label_path, config.train_data_path, config.data_path)
    combine_data(config.test_guid_label_path, config.test_data_path, config.data_path)