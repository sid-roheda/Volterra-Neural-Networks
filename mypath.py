class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/Users/sid.roheda/Downloads/VNN_Code/UCF-101/'

            # Save preprocess data into output_dir
            output_dir = '/Users/sid.roheda/Downloads/VNN_Code/UCF-101/ucf101_pre/'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Volumes/SID_1TB/VNN_data/hmdb51/'

            output_dir = '/Volumes/SID_1TB/VNN_data/hmdb51/hmdb51_pre/'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/home/sroheda/4Dim_VN/pytorch-video-recognition/dataloaders/UCF101/c3d-pretrained.pth'