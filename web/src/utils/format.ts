/**
 * Formats a chapter title from a slug.
 * Removes numeric prefix (e.g., '01-'), language suffixes (e.g. '.zh-cn', '.en'),
 * replaces hyphens with spaces, and capitalizes the first letter.
 * @param {string} slug - The chapter slug.
 * @returns {string} The formatted title.
 */
export function formatChapterTitle(slug: string): string {
    const title = slug.split("/").pop() || "";
    const cleanTitle = title.replace(/[\._-]?(zh-cn|ZH-CN)$|[\._-](en|EN)$/i, '');
    const formattedTitle = cleanTitle.replace(/^\d+-/, "").replace(/-/g, " ");
    if (formattedTitle.length > 0) {
        return formattedTitle.charAt(0).toUpperCase() + formattedTitle.slice(1);
    }
    return formattedTitle;
}

export function formatNumber(num: number): string {
  return (num || 0).toLocaleString("en-US");
}

export function formatBytes(bytes: number): string {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}
