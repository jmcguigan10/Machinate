class Machinate < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinate"
  url "https://github.com/jmcguigan10/Machinate/releases/download/v0.3.0/machinate-0.3.0.tar.gz"
  sha256 "ccd7956ef59f00f824daae16ab8d7a9047af04784d5c0ebc275e8713573901a6"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinate", shell_output("#{bin}/macht --help")
  end
end
