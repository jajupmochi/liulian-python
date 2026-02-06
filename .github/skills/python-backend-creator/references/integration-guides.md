# Integration Guides

## GitHub Copilot

### Setup
1. Add the python-backend-creator skill to your Copilot configuration
2. Open a new chat session in VS Code

### Invoking the Skill
In Copilot Chat, type:
```
@workspace /python-backend-creator I want to create a Python backend for [description]
```

Or simply describe your goal:
```
Help me scaffold a new Python backend project for a data processing pipeline
```

### Using Sub-Prompts with Copilot
When python-backend-creator provides a sub-prompt for a specific file:

1. **Create the target file** in your editor at the specified path
2. **Copy the sub-prompt** into Copilot Chat:
   ```
   Generate the following Python module: [paste sub-prompt content]
   ```
3. **Review** the generated code against the validation criteria listed in the sub-prompt
4. **Confirm** acceptance back in the python-backend-creator conversation

### Tips for Copilot
- Use `@workspace` to give Copilot context about existing files
- For complex modules, paste the sub-prompt's PUBLIC API REQUIREMENTS as a comment block at the top of the file and let Copilot inline-complete the implementation
- After generation, ask Copilot: "Does this code match these requirements: [paste validation criteria]"

---

## Claude Code

### Setup
Add the skill to your Claude Code configuration:
```bash
claude skill add python-backend-creator
```

Or reference the SKILL.md directly:
```bash
claude --skill /path/to/python-backend-creator/SKILL.md
```

### Invoking the Skill
```bash
claude "I want to create a Python backend for a REST API service using FastAPI"
```

Claude Code will automatically recognize the python-backend-creator skill and begin the Stage 0 questionnaire.

### Using Sub-Prompts with Claude Code
Claude Code can execute sub-prompts autonomously:

1. When python-backend-creator presents a sub-prompt, Claude Code will generate the file directly
2. Validation checks run automatically
3. You only need to confirm at each review checkpoint (A/B/C/D/E)

### Tips for Claude Code
- Claude Code handles file creation and validation natively — less manual work needed
- Use `--continue` to resume an interrupted workflow
- For multi-file tasks, Claude Code can maintain context across the full session
- Review the generated files with `claude "review the code in src/mypackage/"` if needed

---

## Cursor

### Setup
1. Open Cursor IDE
2. Add python-backend-creator as a custom skill or load the SKILL.md in your project

### Invoking the Skill
In Cursor's AI chat (Cmd+L / Ctrl+L):
```
I want to use python-backend-creator to scaffold a new Python backend project
```

### Using Sub-Prompts with Cursor
When python-backend-creator provides a sub-prompt:

1. **Open the target file path** using Cursor's file creation (Cmd+N)
2. **Paste the sub-prompt** into the Composer (Cmd+I):
   ```
   Create a Python module following this specification: [paste sub-prompt]
   ```
3. **Use Cursor's diff view** to review the generated code
4. **Accept or reject** the changes, then confirm back in the workflow

### Tips for Cursor
- Use Composer mode for complex file generation — it handles multi-section code better
- Enable "Codebase" context so Cursor can see existing project files
- For validation, ask: "Check this code against: [paste validation criteria]"
- Use Cmd+K for inline edits when making small modifications requested during review

---

## General / Other Agents

For any other LLM-based coding assistant:

1. **Start the workflow** by describing your project goals
2. **Answer the questionnaire** — all 11 questions in Stage 0
3. **Make architecture decisions** — respond to each option with your choice letter
4. **Review critic output** — select Proceed, Revise, or Clarify
5. **When sub-prompts are provided**:
   - Copy the complete sub-prompt text
   - Paste it into your agent's input
   - Collect the generated code
   - Paste it back or confirm it's been created
6. **At review checkpoints** — respond with A (accept), B (modify), C (regenerate), D (defer), or E (abort)

The sub-prompt template is designed to be agent-agnostic. Any agent that can interpret structured instructions and generate Python code will work.
