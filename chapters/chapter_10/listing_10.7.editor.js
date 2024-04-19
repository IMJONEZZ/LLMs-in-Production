// Create inline completion provider, this makes suggestions inline
const provider = {
    provideInlineCompletionItems: async (
            document, position, context, token
        ) => {
        // Grab VSCode editor and selection
        const editor = vscode.window.activeTextEditor;
        const selection = editor.selection;
        const manualKind = 0;
        const manuallyTriggered = context.triggerKind == manualKind;

        // If highlighted back to front, put cursor at the end and rerun
        if (manuallyTriggered && position.isEqual(selection.start)) {
            editor.selection = new vscode.Selection(
                selection.start, selection.end
            );
            vscode.commands.executeCommand(
                "editor.action.inlineSuggest.trigger"
            );
            return []
        }

        // On activation send highlighted text to LLM for suggestions
        if (manuallyTriggered && selection && !selection.isEmpty) {
            // Grab highlighted text
            const selectionRange = new vscode.Range(
                selection.start, selection.end
            );
            const highlighted = editor.document.getText(selectionRange);
            
            // Send highlighted code to LLM
        }
    }
};
