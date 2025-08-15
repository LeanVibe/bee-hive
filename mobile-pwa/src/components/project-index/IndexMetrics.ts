/**
 * Index Metrics Component - Placeholder
 * 
 * Health and performance dashboard for project index metrics.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

@customElement('index-metrics')
export class IndexMetrics extends LitElement {
  @property({ type: String }) declare projectId: string;
  @property({ type: Boolean }) declare compact: boolean;

  static styles = css`
    :host {
      display: block;
      padding: 2rem;
      text-align: center;
      background: var(--surface-color, #ffffff);
      border-radius: 0.5rem;
      border: 1px solid var(--border-color, #e5e7eb);
    }
  `;

  render() {
    return html`
      <h3>Index Metrics</h3>
      <p>Performance metrics dashboard coming soon...</p>
    `;
  }
}