// Import VSCode API library
const vscode = require('vscode');

// This method is called when your extension is activated
function activate(context) {
  console.log('Congratulations, your extension "llm-coding-copilot" is now active!');

  // This creates and registers a new command, matching package.json
  // But we won’t use it!
  let disposable = vscode.commands.registerCommand('llm-coding-copilot.helloWorld', function () {
    // The code you place here will be executed every time your command is executed

    // Display a message box to the user
    vscode.window.showInformationMessage('Hello World from llm_coding_copilot!');
  });

  context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
  activate,
  deactivate
}
