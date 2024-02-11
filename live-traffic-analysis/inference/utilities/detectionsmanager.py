class DetectionsManager:

    def __init__(self) -> None:
        self.trackers = {}

    def update(self, detections, detections_in_zones):
        for zone_id, detections_in_zone in enumerate(detections_in_zones):
            if (
                detections_in_zone is not None
                and detections_in_zone.tracker_id is not None
            ):
                for tracker_id in detections_in_zone.tracker_id:
                    if not tracker_id in self.trackers:
                        self.trackers[tracker_id] = [None, None]
                    if not self.trackers[tracker_id][0]:
                        self.trackers[tracker_id][0] = zone_id
                    elif (
                        not self.trackers[tracker_id][1]
                        and zone_id not in self.trackers[tracker_id]
                    ):
                        self.trackers[tracker_id][1] = zone_id

        print(self.trackers)
