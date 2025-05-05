import os
import csv
import datetime

class AttendanceManager:
    def __init__(self, attendance_dir="attendance_records"):
        self.attendance_dir = attendance_dir
        self.ensure_directory_exists()
        self.recorded_today = set()  # Track who's already been recorded today
        
    def ensure_directory_exists(self):
        if not os.path.exists(self.attendance_dir):
            os.makedirs(self.attendance_dir)
            
    def mark_attendance(self, name):
        # Skip if already marked today or if unknown
        if name in self.recorded_today or name == "Unknown":
            return False
            
        # Get current date and time
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        
        # Create filename for today's attendance
        filename = os.path.join(self.attendance_dir, f"attendance_{date_string}.csv")
        
        # Create file with headers if it doesn't exist
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['Name', 'Time', 'Date'])
                
            writer.writerow([name, time_string, date_string])
            
        # Mark as recorded for today
        self.recorded_today.add(name)
        return True
        
    def get_attendance_report(self, date=None):
        # If no date provided, use today's date
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        filename = os.path.join(self.attendance_dir, f"attendance_{date}.csv")
        
        if not os.path.isfile(filename):
            return []
            
        attendance_list = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                attendance_list.append(row)
                
        return attendance_list
