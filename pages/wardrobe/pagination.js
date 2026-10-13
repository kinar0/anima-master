// Shared by the editable library and the translation review table.
// Keep paging state in memory: AstrBot's sandboxed page may prohibit storage.
window.WardrobePagination = class {
  constructor(containers, onChange) {
    this.scopes = new Map();
    this.scope = "default";
    this.total = 0;
    this.onChange = onChange;
    this.controls = containers.map((container) => this.mount(container));
  }

  get current() {
    if (!this.scopes.has(this.scope)) this.scopes.set(this.scope, { page: 1, size: 30 });
    return this.scopes.get(this.scope);
  }

  get pages() { return Math.max(1, Math.ceil(this.total / this.current.size)); }

  mount(container) {
    const create = (tag, className, text) => {
      const element = document.createElement(tag);
      element.className = className;
      if (text !== undefined) element.textContent = text;
      return element;
    };
    container.classList.add("pagination");
    container.setAttribute("aria-label", "列表分页");
    const summary = create("span", "pagination-summary");
    summary.setAttribute("role", "status");
    const actions = create("div", "pagination-actions");
    const sizeLabel = create("label", "pagination-size", "每页 ");
    const size = create("select", "page-size");
    [10, 30, 50, 100, 200].forEach((value) => {
      const option = create("option", "", `${value} 条`);
      option.value = String(value);
      size.append(option);
    });
    size.setAttribute("aria-label", "每页显示数量");
    size.addEventListener("change", () => {
      const nextSize = Number(size.value);
      if (![10, 30, 50, 100, 200].includes(nextSize)) return;
      const first = (this.current.page - 1) * this.current.size;
      this.current.size = nextSize;
      this.current.page = Math.floor(first / nextSize) + 1;
      this.onChange();
    });
    sizeLabel.append(size);
    const buttons = {};
    [["first", "首页", () => 1], ["prev", "上一页", () => this.current.page - 1],
      ["next", "下一页", () => this.current.page + 1], ["last", "末页", () => this.pages]]
      .forEach(([key, text, target]) => {
        const button = create("button", `button ghost page-${key}`, text);
        button.type = "button";
        button.addEventListener("click", () => this.go(target()));
        buttons[key] = button;
      });
    const jumpLabel = create("label", "pagination-jump", "第 ");
    const jump = create("input", "page-number");
    jump.type = "number";
    jump.min = "1";
    jump.step = "1";
    jump.setAttribute("aria-label", "跳转页码");
    const jumpTo = () => {
      if (!jump.value.trim() || !this.go(Number(jump.value))) jump.value = String(this.current.page);
    };
    jump.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); jumpTo(); }
    });
    jumpLabel.append(jump, create("span", "", " 页"));
    const submit = create("button", "button soft page-go", "跳转");
    submit.type = "button";
    submit.addEventListener("click", jumpTo);
    actions.append(sizeLabel, buttons.first, buttons.prev, buttons.next, buttons.last, jumpLabel, submit);
    container.append(summary, actions);
    return { summary, size, jump, submit, ...buttons };
  }

  go(page) {
    if (!Number.isFinite(page) || !Number.isInteger(page)) return false;
    this.current.page = Math.max(1, Math.min(this.pages, page));
    this.onChange();
    return true;
  }

  reset() { this.current.page = 1; }

  update(total, allTotal = total) {
    this.total = total;
    this.current.page = Math.max(1, Math.min(this.pages, this.current.page));
    const start = (this.current.page - 1) * this.current.size;
    const end = Math.min(total, start + this.current.size);
    const summary = `${total === allTotal ? `共 ${total} 条` : `筛选 ${total} / ${allTotal} 条`} · 显示 ${total ? start + 1 : 0}–${end} 条 · 第 ${this.current.page} / ${this.pages} 页`;
    this.controls.forEach((control) => {
      control.summary.textContent = summary;
      control.size.value = String(this.current.size);
      control.jump.value = String(this.current.page);
      control.jump.max = String(this.pages);
      control.first.disabled = control.prev.disabled = this.current.page === 1;
      control.last.disabled = control.next.disabled = this.current.page === this.pages;
      control.jump.disabled = control.submit.disabled = total === 0;
    });
    return { start, end };
  }
};
