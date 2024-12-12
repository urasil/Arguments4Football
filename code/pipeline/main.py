from identify_arguments import ArgumentIdentification    
from cluster_arguments import Clustering
from topic_extraction import ExtractTopic
from collections import defaultdict
class Main:
    def __init__(self):
        model_name = "../curr-pure-best"
        self.identifier = ArgumentIdentification(model_name) # "https://www.skysports.com/football/newcastle-united-vs-west-ham-united/report/505925"
        self.cluster = Clustering()
        # self.cluster.group_labels(self.cluster.cluster())
        self.extract = ExtractTopic()
        self.topic_to_args = defaultdict(list)
    
    def cluster_and_label_args(self, url):
        identified_args = self.identifier.identify_arguments(url)
        cluster_labels = self.cluster.group_labels(self.cluster.cluster("kmeans", identified_args))
        temp_args = []

        for clus, args_idxs in cluster_labels.items():
            for arg_idx in args_idxs:
                temp_args.append(identified_args[arg_idx])
            topic = self.extract.determine_topic(temp_args)
            print(topic)
            self.topic_to_args[topic] = temp_args
            temp_args = []
        return self.topic_to_args
if __name__ == "__main__":
    main = Main()
    print(main.cluster_and_label_args("https://www.skysports.com/football/newcastle-united-vs-west-ham-united/report/505925"))