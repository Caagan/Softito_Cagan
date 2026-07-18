import random
import datetime
import csv
import os

LEVELS = ["INFO", "WARNING", "ERROR", "DEBUG"]
SERVICES = ["auth-service", "payment-service", "order-service", "user-service", "notification-service"]
MESSAGES = {
    "INFO": [
        "Request processed successfully",
        "User login successful",
        "Payment completed",
        "Order placed",
        "Cache refreshed",
        "Health check passed",
        "Database connection established",
    ],
    "WARNING": [
        "High memory usage detected",
        "Slow query detected",
        "Rate limit approaching",
        "Disk space low",
        "Connection pool exhausted",
    ],
    "ERROR": [
        "Database connection failed",
        "Payment gateway timeout",
        "Authentication failed",
        "Null pointer exception",
        "Out of memory error",
    ],
    "DEBUG": [
        "Entering function process_order",
        "Query execution time: 234ms",
        "Response payload size: 1.2KB",
        "Thread pool status: 8/16 active",
    ],
}

def generate_log_line(timestamp):
    level = random.choices(LEVELS, weights=[50, 25, 15, 10])[0]
    service = random.choice(SERVICES)
    message = random.choice(MESSAGES[level])
    ip = f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"{timestamp} [{level}] [{service}] {message} ip={ip}"

def generate_logs(output_path="server_logs.csv", num_lines=5000):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "level", "service", "message", "ip"])

        for i in range(num_lines):
            ts = start + datetime.timedelta(seconds=random.randint(0, 86400 * 30))
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            line = generate_log_line(ts_str)
            parts = line.split(" ", 4)
            level = parts[1].strip("[]")
            service = parts[2].strip("[]")
            rest = parts[3].split(" ip=")
            message = rest[0]
            ip = rest[1] if len(rest) > 1 else "0.0.0.0"
            writer.writerow([ts_str, level, service, message, ip])

    print(f"    {num_lines} log satiri olusturuldu: {output_path}")

if __name__ == "__main__":
    print("=" * 50)
    print("  LOG GENERATOR — BigData icin log verisi uretme")
    print("=" * 50)
    output = r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\BigData\big-data-log-analytics\server_logs.csv"
    generate_logs(output, num_lines=5000)
    print("    Tamamlandi!")
