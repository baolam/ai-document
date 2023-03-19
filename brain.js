const notifier = require("node-notifier");
const cmd = require("node-cmd");
const chokidar = require("chokidar");
const EventEmitter = require("events").EventEmitter;

class Observe extends EventEmitter {
	constructor()
	{
		super();
		this.watchers = [];
	}

	watchFolder(targetFolder)
	{
		let my_watch = chokidar.watch(targetFolder, { persistent : true });
		my_watch.on("change", (path) => {
			notifier.notify("Tiến hành cập nhật kết quả lên git");
			/// Tiến hành đồng bộ kết quả (lắng nghe các folder thêm)
			cmd.runSync("git add .");
			/// Tiến hành commit
			cmd.runSync("git commit -am 'updating automatically from service' ");
			/// Tiến hành remote đến target
			cmd.runSync("git remote add origin https://github.com/baolam/ai-document.git");
		});
	}
}

module.exports = Observe;