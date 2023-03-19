const notifier = require("node-notifier");
const cmd = require("node-cmd");
const chokidar = require("chokidar");
const EventEmitter = require("events").EventEmitter;

class Observe extends EventEmitter {
	timeout = 15;

	constructor()
	{
		super();
		this.watchers = [];

		notifier.on("click", () => {
			/// Tiến hành đồng bộ kết quả (lắng nghe các folder thêm)
			console.log("Lệnh 1 ");
			let r = cmd.runSync("git add .");
			console.log(r.data);
			/// Tiến hành commit
			console.log("Lệnh 2 ");
			r = cmd.runSync('git commit -am "updating automatically from service" ');
			console.log(r.data);
			/// Tiến hành remote đến target
			console.log("Lệnh 3");
			cmd.runSync("git remote add origin https://github.com/baolam/ai-document.git");
			console.log(r.data);
			/// Tiến hành push
			console.log("Lệnh 4");
			cmd.runSync("git push origin main");
			console.log(r.data);
			/// Thông báo đến người dùng
			notifier.notify("Cập nhật dữ liệu thành công");
		})
	}

	watchFolder(targetFolder)
	{
		let my_watch = chokidar.watch(targetFolder, { persistent : true });
		my_watch.on("change", (_path) => {
			notifier.notify({
				title : "Từ dịch vụ cập nhật kết quả lên git",
				message : "Nhấn thẳng vào thông báo để tiến hành cập nhật. Nếu không muốn hãy nhấn chính xác vào dấu x hoặc đợi sau " + this.timeout + " giây.",
				time : this.timeout * 1000,
				sound : true,
				wait : true
			});
		});
	}
}

module.exports = Observe;